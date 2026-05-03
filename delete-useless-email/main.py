from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

import imaplib, email, os, json, re, datetime, sqlite3, asyncio, ssl, time
from email.header import decode_header, make_header
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from html.parser import HTMLParser
from pathlib import Path

load_dotenv()


# -------------------------------------------------
# CONFIG
# -------------------------------------------------


EMAIL_LIMIT      = 8000   # total emails to process
BATCH_SIZE       = 30     # emails per LLM call
IMAP_WORKERS     = 5      # parallel IMAP threads — keep low to avoid server rate limits
FETCH_QUEUE_MAX  = 5      # backpressure: max batches waiting for LLM
DELETE_QUEUE_MAX = 20     # backpressure: max delete jobs queued
CACHE_PATH       = Path("email_cache.db")
RETRY_ATTEMPTS   = 4      # max retries for IMAP operations
RETRY_BASE_DELAY = 2.0    # seconds — doubles on each attempt: 2, 4, 8, 16
DRY_RUN          = False  # True = classify only, nothing deleted
SENTINEL         = None   # end-of-queue signal


# -------------------------------------------------
# 1. HELPERS
# -------------------------------------------------

def _strip_surrogates(s: str) -> str:
    """Replace lone UTF-16 surrogates that can't be encoded as UTF-8."""
    return s.encode("utf-8", errors="replace").decode("utf-8")


def decode_mime(value: str) -> str:
    if not value:
        return ""
    try:
        parts = decode_header(value)
        safe_parts = []
        for raw, charset in parts:
            if isinstance(raw, bytes):
                safe_parts.append((raw, charset or "utf-8"))
            else:
                safe_parts.append((raw, charset))
        decoded = str(make_header(safe_parts))
    except (UnicodeDecodeError, LookupError):
        # Fallback: decode each chunk independently, replacing bad bytes
        parts = decode_header(value)
        chunks = []
        for raw, charset in parts:
            if isinstance(raw, bytes):
                chunks.append(raw.decode(charset or "utf-8", errors="replace"))
            else:
                chunks.append(raw)
        decoded = "".join(chunks)
    return _strip_surrogates(decoded)


def get_imap() -> imaplib.IMAP4_SSL:
    mail = imaplib.IMAP4_SSL(os.getenv("IMAP_SERVER"))
    mail.login(os.getenv("EMAIL"), os.getenv("PASSWORD"))
    return mail


def get_imap_with_retry() -> imaplib.IMAP4_SSL:
    """Open IMAP connection, retrying on SSL/EOF errors with exponential backoff."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return get_imap()
        except (ssl.SSLError, imaplib.IMAP4.abort, OSError) as ex:
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            wait = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"   ⚠️  IMAP connect error (attempt {attempt + 1}/{RETRY_ATTEMPTS}): {ex}")
            print(f"       Retrying in {wait:.0f}s...")
            time.sleep(wait)


def safe_logout(mail):
    """Silently close an IMAP session — never raises."""
    try:
        mail.logout()
    except Exception:
        pass


def is_gmail(mail: imaplib.IMAP4_SSL) -> bool:
    """Detect Gmail by checking for the X-GM-EXT-1 IMAP capability."""
    try:
        _, caps = mail.capability()
        return b"X-GM-EXT-1" in caps[0]
    except Exception:
        return False


class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        if tag in ("p", "br", "tr", "div", "li"):
            self.text_parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            s = data.strip()
            if s:
                self.text_parts.append(s)

    def get_text(self):
        return " ".join(self.text_parts)


def html_to_text(html: str) -> str:
    p = HTMLTextExtractor()
    p.feed(html)
    return p.get_text()


# -------------------------------------------------
# 2. CACHE (SQLite)
# -------------------------------------------------

def init_cache() -> sqlite3.Connection:
    conn = sqlite3.connect(CACHE_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            uid        TEXT PRIMARY KEY,
            from_      TEXT,
            subject    TEXT,
            date       TEXT,
            body       TEXT,
            fetched_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS classifications (
            uid           TEXT PRIMARY KEY,
            category      TEXT,
            reason        TEXT,
            subject       TEXT,
            classified_at TEXT
        )
    """)
    conn.commit()
    return conn


def cache_get(conn: sqlite3.Connection, uids: list[str]) -> dict[str, dict]:
    if not uids:
        return {}
    placeholders = ",".join("?" * len(uids))
    rows = conn.execute(
        f"SELECT uid, from_, subject, date, body FROM emails WHERE uid IN ({placeholders})",
        uids
    ).fetchall()
    return {
        row[0]: {"id": row[0], "from": row[1], "subject": row[2],
                 "date": row[3], "body": row[4]}
        for row in rows
    }


def cache_save_many(conn: sqlite3.Connection, records: list[dict]):
    conn.executemany(
        """INSERT OR REPLACE INTO emails (uid, from_, subject, date, body, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [
            (r["id"], r["from"], r["subject"], r.get("date", ""),
             r.get("body", ""), datetime.datetime.now().isoformat())
            for r in records
        ]
    )
    conn.commit()


def cache_evict_deleted(conn: sqlite3.Connection, live_uids: set[str]):
    cached = {row[0] for row in conn.execute("SELECT uid FROM emails").fetchall()}
    stale  = cached - live_uids
    if stale:
        conn.executemany("DELETE FROM emails WHERE uid = ?", [(u,) for u in stale])
        conn.commit()
        print(f"🧹 Evicted {len(stale)} stale UIDs from cache.")


def cache_get_classifications(conn: sqlite3.Connection, uids: list[str]) -> dict[str, dict]:
    if not uids:
        return {}
    placeholders = ",".join("?" * len(uids))
    rows = conn.execute(
        f"SELECT uid, category, reason, subject FROM classifications WHERE uid IN ({placeholders})",
        uids
    ).fetchall()
    return {row[0]: {"id": row[0], "category": row[1], "reason": row[2], "subject": row[3]}
            for row in rows}


def _safe_str(s) -> str:
    if isinstance(s, list):
        s = s[0] if s else ""
    if not s:
        return s
    return s.encode("utf-8", "surrogatepass").decode("utf-8", "replace")


def cache_save_classifications(conn: sqlite3.Connection, results: list[dict]):
    conn.executemany(
        """INSERT OR REPLACE INTO classifications (uid, category, reason, subject, classified_at)
           VALUES (?, ?, ?, ?, ?)""",
        [
            (r["id"], r.get("category", "REVIEW"), _safe_str(r.get("reason", "")),
             _safe_str(r.get("subject", "")), datetime.datetime.now().isoformat())
            for r in results
        ]
    )
    conn.commit()


# -------------------------------------------------
# 3. RULE-BASED PRE-FILTER
# -------------------------------------------------

RULE_DELETE_SENDER_KEYWORDS = {
    "no-reply", "noreply", "mailer-daemon", "newsletter", "notifications",
    "do-not-reply", "donotreply", "bounce", "alert", "update", "offers",
    "promo", "deals", "marketing", "news", "info@", "support@", "team@",
    "hello@", "hi@", "contact@"
}
RULE_DELETE_DOMAINS = {
    "mailchimp.com", "sendgrid.net", "klaviyo.com", "constantcontact.com",
    "mailgun.org", "amazonses.com", "em.shopify.com", "e.medium.com",
    "notify.twitter.com", "facebookmail.com", "linkedin.com",
}
RULE_DELETE_SUBJECT_KEYWORDS = {
    "unsubscribe", "% off", "sale ends", "limited time", "special offer",
    "verify your email", "confirm your email", "confirm your account",
    "your order has", "your package", "shipping update", "has been shipped",
    "your password", "reset your password", "sign in attempt", "login attempt",
    "weekly digest", "monthly digest", "newsletter", "new follower",
    "liked your", "commented on", "mentioned you",
}
RULE_KEEP_SENDER_DOMAINS = {
    "gmail.com", "outlook.com", "hotmail.com", "yahoo.com", "icloud.com"
}


def rule_based_classify(e: dict) -> str | None:
    """Returns 'DELETE'/'KEEP' if confident, None if LLM should decide."""
    sender  = e.get("from", "").lower()
    subject = e.get("subject", "").lower()
    match   = re.search(r"@([\w.\-]+)", sender)
    domain  = match.group(1) if match else ""

    if domain in RULE_KEEP_SENDER_DOMAINS:             return "KEEP"
    if any(d in domain for d in RULE_DELETE_DOMAINS):  return "DELETE"
    if any(kw in sender  for kw in RULE_DELETE_SENDER_KEYWORDS):  return "DELETE"
    if any(kw in subject for kw in RULE_DELETE_SUBJECT_KEYWORDS): return "DELETE"
    return None


# -------------------------------------------------
# 4. IMAP FETCHERS
# -------------------------------------------------

_HEADER_CHUNK = 500  # UIDs per bulk RFC822.HEADER fetch command


def _parse_body(msg) -> str:
    """Extract plain text from a parsed email.Message object."""
    plain_body, html_body = "", ""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if part.get("Content-Disposition", "").startswith("attachment"):
                continue
            if ct == "text/plain" and not plain_body:
                plain_body = _strip_surrogates(part.get_payload(decode=True).decode(errors="replace"))
            elif ct == "text/html" and not html_body:
                html_body = _strip_surrogates(part.get_payload(decode=True).decode(errors="replace"))
    else:
        raw = _strip_surrogates(msg.get_payload(decode=True).decode(errors="replace"))
        if msg.get_content_type() == "text/plain":
            plain_body = raw
        else:
            html_body = raw

    if plain_body.strip():
        return plain_body.strip()[:500]
    if html_body.strip():
        return html_to_text(html_body).strip()[:500]
    return "[ERROR] No readable body."


def fetch_all_headers(limit: int, folder: str = "INBOX") -> list[dict]:
    """
    Fetch headers for up to `limit` emails.

    Old: one UID FETCH per email = N round-trips.
    New: one UID FETCH per 500 emails = ceil(N/500) round-trips.
    The server MUST include the UID in UID FETCH responses (RFC 3501),
    so we parse it from the metadata instead of trusting list order.
    """
    mail = get_imap_with_retry()
    mail.select(folder, readonly=True)
    _, data = mail.uid("search", None, "ALL")
    ids   = data[0].split()[-limit:]
    total = len(ids)
    results = []

    for i in range(0, total, _HEADER_CHUNK):
        chunk   = ids[i : i + _HEADER_CHUNK]
        uid_set = b",".join(chunk)
        _, msg_data = mail.uid("fetch", uid_set, "(UID RFC822.HEADER)")

        for item in msg_data:
            if not isinstance(item, tuple) or len(item) < 2:
                continue
            uid_match = re.search(rb'\bUID\s+(\d+)', item[0])
            if not uid_match:
                continue
            uid = uid_match.group(1).decode()
            msg = email.message_from_bytes(item[1])
            results.append({
                "id":      uid,
                "from":    decode_mime(msg["From"]),
                "subject": decode_mime(msg["Subject"]),
                "date":    msg["Date"],
            })

        fetched = min(i + _HEADER_CHUNK, total)
        print(f"   📨 Headers: {fetched}/{total} "
              f"({len(chunk)} fetched in 1 IMAP command)")

    safe_logout(mail)
    return results


def _fetch_bodies_worker(uids: list[str], folder: str = "INBOX") -> list[tuple[str, str]]:
    """
    Fetch bodies for a list of UIDs using a single IMAP connection.

    Old: one connection + one FETCH per UID.
    New: one connection + one bulk FETCH for all UIDs assigned to this worker.
    """
    last_error = None

    for attempt in range(RETRY_ATTEMPTS):
        mail = None
        try:
            mail = get_imap_with_retry()
            mail.select(folder, readonly=True)
            uid_set = b",".join(uid.encode() for uid in uids)
            _, msg_data = mail.uid("fetch", uid_set, "(UID RFC822)")
            safe_logout(mail)

            results = []
            for item in msg_data:
                if not isinstance(item, tuple) or len(item) < 2:
                    continue
                uid_match = re.search(rb'\bUID\s+(\d+)', item[0])
                if not uid_match:
                    continue
                uid = uid_match.group(1).decode()
                msg = email.message_from_bytes(item[1])
                results.append((uid, _parse_body(msg)))

            # Any UID missing from the response gets an error placeholder
            returned = {uid for uid, _ in results}
            for uid in uids:
                if uid not in returned:
                    results.append((uid, "[ERROR] No data returned."))

            return results

        except (ssl.SSLError, imaplib.IMAP4.abort, OSError) as ex:
            last_error = ex
            safe_logout(mail) if mail else None
            if attempt < RETRY_ATTEMPTS - 1:
                wait = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"   ⚠️  Body fetch retry {attempt + 1}/{RETRY_ATTEMPTS - 1} "
                      f"({len(uids)} UIDs) in {wait:.0f}s: {ex}")
                time.sleep(wait)
        except Exception as ex:
            safe_logout(mail) if mail else None
            return [(uid, f"[ERROR] {ex}") for uid in uids]

    return [(uid, f"[ERROR] Failed after {RETRY_ATTEMPTS} attempts: {last_error}")
            for uid in uids]


# -------------------------------------------------
# 5. OUTPUT PARSER
# -------------------------------------------------

def sanitize(results: list) -> list:
    FAKE_IDS      = (None, "unknown", "[each email ID in list]", "")
    FAKE_SUBJECTS = (None, "unknown", "[each subject in list]", "")
    sanitized = []
    for item in results:
        if "id" in item:
            item["id"] = str(item["id"])
        if item.get("id")      in FAKE_IDS:      continue
        if item.get("subject") in FAKE_SUBJECTS: continue
        for k in ("folder", "limit", "email_id", "type", "sender_domain", "body"):
            item.pop(k, None)
        item.setdefault("category", "REVIEW")
        item.setdefault("reason",   "no reason provided")
        sanitized.append(item)
    return sanitized


def parse_llm_output(raw: str) -> list:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        r = json.loads(cleaned)
        if isinstance(r, list):
            return sanitize(r)
    except json.JSONDecodeError:
        pass
    objs = re.findall(r'\{[^{}]+\}', cleaned)
    results = []
    for o in objs:
        try:
            results.append(json.loads(o))
        except json.JSONDecodeError:
            continue
    return sanitize(results) if results else []


# -------------------------------------------------
# 6. LLM
# -------------------------------------------------

CLASSIFIER_SYSTEM_PROMPT = """You are an email classifier. Classify every email — never skip one.

Categories:
- DELETE: promotional, newsletters, spam, marketing, verification codes, password resets,
  order confirmations, shipping updates, social media notifications, non-personal emails.
- KEEP: emails from real people, receipts, invoices, work emails (except LinkedIn), interviews, bills.
- REVIEW: only if truly impossible to decide.

Rules:
- Respond with ONLY a valid JSON array. No markdown, no extra text.
- "reason" is REQUIRED for every item.
- gmail.com/outlook.com sender → likely KEEP. no-reply/newsletter domains → likely DELETE.
- Prefer DELETE over REVIEW for promotional content.
- Prefer KEEP over REVIEW when sender looks like a real person.
- If body has [ERROR], classify from subject + sender only.
- REVIEW is a last resort."""


llm = ChatOpenAI(
    model=os.getenv("MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ"),
    base_url=os.getenv("LLM_BASE_URL", ""),
    api_key=os.getenv("LLM_API_KEY", ""),
)


# -------------------------------------------------
# 7. DELETE HELPERS
# -------------------------------------------------

_IMAP_UID_CHUNK = 500  # max UIDs per single STORE command (avoids server line-length limits)


def _delete_all_sync(to_delete: list) -> tuple[int, str | None]:
    """
    Single IMAP connection for ALL deletions.

    Old approach: one new SSL connection + one STORE per email per batch.
    New approach: one SSL connection total + chunked multi-UID STORE commands
                  (e.g. "UID STORE 1,2,3,...,500 +FLAGS \\Deleted") + one EXPUNGE.

    This cuts N*handshake overhead + N round-trips down to
    1 handshake + ceil(N/500) round-trips.
    """
    mail = None
    try:
        mail = get_imap_with_retry()
        gmail = is_gmail(mail)
        server_type = "Gmail" if gmail else "Standard IMAP"
        print(f"   🔌 Connected ({server_type}), selecting INBOX "
              f"({len(to_delete)} emails to delete)...")

        typ, data = mail.select("INBOX", readonly=False)
        if typ != "OK":
            return 0, f"SELECT INBOX failed: {data}"

        succeeded = 0
        for i in range(0, len(to_delete), _IMAP_UID_CHUNK):
            chunk    = to_delete[i : i + _IMAP_UID_CHUNK]
            uid_set  = b",".join(str(e["id"]).strip().encode() for e in chunk)

            if gmail:
                typ, data = mail.uid("STORE", uid_set, "+X-GM-LABELS", "\\Trash")
            else:
                typ, data = mail.uid("STORE", uid_set, "+FLAGS", "\\Deleted")

            if typ == "OK":
                succeeded += len(chunk)
                print(f"      🗑️  Marked {len(chunk)} UIDs "
                      f"(chunk {i // _IMAP_UID_CHUNK + 1}/"
                      f"{(len(to_delete) - 1) // _IMAP_UID_CHUNK + 1})")
            else:
                print(f"      ⚠️  STORE failed for chunk at offset {i}: {data}")

        typ, data = mail.expunge()
        if typ != "OK":
            return succeeded, f"EXPUNGE failed: {data}"

        safe_logout(mail)
        return succeeded, None

    except (ssl.SSLError, imaplib.IMAP4.abort, OSError) as ex:
        safe_logout(mail) if mail else None
        return 0, str(ex)
    except Exception as ex:
        safe_logout(mail) if mail else None
        return 0, str(ex)


async def _delete_all_with_retry(to_delete: list) -> int:
    for attempt in range(RETRY_ATTEMPTS):
        count, error = await asyncio.to_thread(_delete_all_sync, to_delete)
        if error is None:
            return count
        if attempt == RETRY_ATTEMPTS - 1:
            print(f"   [ERROR] Delete failed after {RETRY_ATTEMPTS} attempts: {error}")
            return 0
        wait = RETRY_BASE_DELAY * (2 ** attempt)
        print(f"   ⚠️  Delete error, retry {attempt + 1}/{RETRY_ATTEMPTS - 1} "
              f"in {wait:.0f}s: {error}")
        await asyncio.sleep(wait)
    return 0


# -------------------------------------------------
# 8. PIPELINE STAGES
# -------------------------------------------------

async def stage_fetch(
    all_headers: list[dict],
    conn: sqlite3.Connection,
    classify_queue: asyncio.Queue,
    executor: ThreadPoolExecutor,
    loop: asyncio.AbstractEventLoop,
):
    """
    Stage 1 — Fetcher
    1. Rule-based filter (no IMAP, no LLM).
    2. Cache lookup — skip IMAP for already-fetched emails.
    3. Parallel IMAP fetch (with retry) for cache misses only.
    4. Save fresh bodies to cache.
    5. Push ready batches onto classify_queue.
    """
    rule_results = []
    needs_llm    = []

    for e in all_headers:
        decision = rule_based_classify(e)
        if decision:
            rule_results.append({
                "id": e["id"], "subject": e["subject"],
                "category": decision, "reason": "rule-based (no LLM needed)",
            })
        else:
            needs_llm.append(e)

    print(f"\n⚡ Pre-filter: {len(rule_results)} rule-classified "
          f"({sum(1 for e in rule_results if e['category'] == 'DELETE')} DELETE / "
          f"{sum(1 for e in rule_results if e['category'] == 'KEEP')} KEEP), "
          f"{len(needs_llm)} going to LLM.")

    if rule_results:
        await classify_queue.put(("rule", rule_results))

    if not needs_llm:
        await classify_queue.put(SENTINEL)
        return

    # --- Cache lookup ---
    uids_for_llm   = [e["id"] for e in needs_llm]
    cached_records = cache_get(conn, uids_for_llm)
    cache_hits     = [uid for uid in uids_for_llm if uid     in cached_records]
    cache_misses   = [uid for uid in uids_for_llm if uid not in cached_records]

    print(f"📦 Cache: {len(cache_hits)} hits (skipping IMAP), "
          f"{len(cache_misses)} misses (fetching from server).\n")

    uid_to_meta = {e["id"]: e for e in needs_llm}

    # --- Fetch only misses from IMAP in parallel sub-batches ---
    if cache_misses:
        total_misses  = len(cache_misses)
        fetched_count = 0

        for i in range(0, total_misses, BATCH_SIZE):
            sub_uids = cache_misses[i : i + BATCH_SIZE]

            # Divide sub-batch among IMAP_WORKERS — each worker gets a slice
            # and fetches all its UIDs in ONE connection (one bulk FETCH command).
            # Old: len(sub_uids) connections.  New: min(IMAP_WORKERS, len) connections.
            n_workers  = min(IMAP_WORKERS, len(sub_uids))
            per_worker = -(-len(sub_uids) // n_workers)  # ceiling division
            chunks     = [sub_uids[j : j + per_worker]
                          for j in range(0, len(sub_uids), per_worker)]
            futures    = [
                loop.run_in_executor(executor, _fetch_bodies_worker, chunk)
                for chunk in chunks
            ]
            worker_results = await asyncio.gather(*futures)
            fetch_results  = [(uid, body)
                              for worker in worker_results for uid, body in worker]

            new_records = []
            for uid, body in fetch_results:
                fetched_count += 1
                meta = uid_to_meta[uid]
                new_records.append({**meta, "body": body})
                print(f"   📩 Body [{fetched_count}/{total_misses}] "
                      f"UID {uid} | {len(body)} chars | "
                      f"{meta['subject'][:45]!r} from {meta['from'][:35]!r}")

            cache_save_many(conn, new_records)

            enriched = [(uid_to_meta[uid], body) for uid, body in fetch_results]
            await classify_queue.put(("llm", enriched))

    # --- Push cache-hit batches ---
    hit_meta = [uid_to_meta[uid] for uid in cache_hits]
    for i in range(0, len(hit_meta), BATCH_SIZE):
        sub      = hit_meta[i : i + BATCH_SIZE]
        enriched = [(e, cached_records[e["id"]]["body"]) for e in sub]
        await classify_queue.put(("llm", enriched))

    await classify_queue.put(SENTINEL)
    print("\n✅ Stage 1 (Fetch) complete.")


async def stage_classify(
    classify_queue: asyncio.Queue,
    delete_queue: asyncio.Queue,
    conn: sqlite3.Connection,
):
    """
    Stage 2 — Classifier
    'rule' batches pass through without LLM.
    'llm' batches check the classification cache first; only uncached
    emails are sent to the LLM. Results are saved back to the cache.
    """
    all_classified = []
    batch_num      = 0

    while True:
        item = await classify_queue.get()
        classify_queue.task_done()

        if item is SENTINEL:
            break

        tag, batch = item

        if tag == "rule":
            all_classified.extend(batch)
            await delete_queue.put(batch)
            continue

        batch_num += 1

        # --- Classification cache lookup ---
        uids        = [e["id"] for e, _ in batch]
        cached_cls  = cache_get_classifications(conn, uids)
        cache_hits  = [cached_cls[uid] for uid in uids if uid in cached_cls]
        needs_llm   = [(e, body) for e, body in batch if e["id"] not in cached_cls]

        if cache_hits:
            print(f"\n⚙️  Batch {batch_num}: {len(cache_hits)} from classification cache, "
                  f"{len(needs_llm)} need LLM.")

        classified_this_batch = list(cache_hits)

        if needs_llm:
            print(f"\n⚙️  LLM batch {batch_num} — {len(needs_llm)} emails...")

            batch_text = "\n\n".join(
                f'ID: {e["id"]}\nFrom: {e["from"]}\nSubject: {e["subject"]}\nBody: {body}'
                for e, body in needs_llm
            )
            prompt = (
                f"Classify these {len(needs_llm)} emails. "
                f"Return ONLY a JSON array with fields: id, subject, category, reason.\n\n"
                f"{batch_text}"
            )

            response = await asyncio.to_thread(
                llm.invoke,
                [SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT), HumanMessage(content=prompt)],
            )

            llm_results = parse_llm_output(response.content)
            cache_save_classifications(conn, llm_results)
            classified_this_batch.extend(llm_results)

        all_classified.extend(classified_this_batch)

        keep   = sum(1 for e in classified_this_batch if e.get("category") == "KEEP")
        delete = sum(1 for e in classified_this_batch if e.get("category") == "DELETE")
        review = sum(1 for e in classified_this_batch if e.get("category") == "REVIEW")
        print(f"   ✅ {len(classified_this_batch)} classified → "
              f"{delete} DELETE / {keep} KEEP / {review} REVIEW "
              f"(total so far: {len(all_classified)})")

        await delete_queue.put(classified_this_batch)

    await delete_queue.put(SENTINEL)
    print("\n✅ Stage 2 (Classify) complete.")


async def stage_delete(
    delete_queue: asyncio.Queue,
    output_collector: list,
):
    """
    Stage 3 — Deleter
    Collects all classified results from the queue, then issues a SINGLE
    bulk IMAP delete (one SSL connection, chunked multi-UID STORE, one EXPUNGE)
    instead of one connection + one STORE per email per batch.
    """
    all_to_delete = []

    if DRY_RUN:
        print("\n🔍 DRY RUN enabled — no emails will be deleted.\n")

    while True:
        batch = await delete_queue.get()
        delete_queue.task_done()

        if batch is SENTINEL:
            break

        output_collector.extend(batch)
        to_delete  = [e for e in batch if e.get("category") == "DELETE"]
        keep_count = sum(1 for e in batch if e.get("category") == "KEEP")
        print(f"   📥 Queued {len(to_delete)} DELETE / {keep_count} KEEP / "
              f"{len(batch) - len(to_delete) - keep_count} REVIEW "
              f"(delete total so far: {len(all_to_delete) + len(to_delete)})")
        all_to_delete.extend(to_delete)

    if not all_to_delete:
        print("\n✅ Stage 3 (Delete) complete — nothing to delete.")
        return

    if DRY_RUN:
        for e in all_to_delete:
            print(f"   🔍 [DRY RUN] Would delete UID {e['id']}: "
                  f"{e.get('subject', '')[:60]!r}")
        print(f"\n✅ Stage 3 (Delete) complete — [DRY RUN] Would have deleted: {len(all_to_delete)}")
        return

    print(f"\n   ⏳ Deleting {len(all_to_delete)} emails in one IMAP session...")
    deleted = await _delete_all_with_retry(all_to_delete)
    print(f"\n✅ Stage 3 (Delete) complete — {deleted}/{len(all_to_delete)} deleted.")


# -------------------------------------------------
# 9. MAIN
# -------------------------------------------------

async def run_pipeline():
    start = datetime.datetime.now()
    print("=" * 60)
    print(f"▶️  Started at: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    if DRY_RUN:
        print("🔍 DRY RUN MODE — no emails will be deleted")
    print("=" * 60)

    # Init cache
    conn         = init_cache()
    cache_size   = CACHE_PATH.stat().st_size / 1024 if CACHE_PATH.exists() else 0
    cached_count = conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
    print(f"\n💾 Cache: {CACHE_PATH} — {cached_count} emails stored ({cache_size:.1f} KB)")

    # Fetch all headers (lightweight — headers only, always fresh from IMAP)
    print(f"\n📬 Fetching headers for up to {EMAIL_LIMIT} emails from INBOX...")
    loop        = asyncio.get_event_loop()
    executor    = ThreadPoolExecutor(max_workers=IMAP_WORKERS)
    all_headers = await loop.run_in_executor(executor, fetch_all_headers, EMAIL_LIMIT)
    print(f"\n✅ Got {len(all_headers)} headers total.")

    # Evict deleted emails from cache
    live_uids = {e["id"] for e in all_headers}
    cache_evict_deleted(conn, live_uids)

    # Build queues with backpressure
    classify_queue = asyncio.Queue(maxsize=FETCH_QUEUE_MAX)
    delete_queue   = asyncio.Queue(maxsize=DELETE_QUEUE_MAX)
    all_classified = []

    print("\n🚀 Starting pipeline (Fetch → Classify → Delete running concurrently)...\n")

    await asyncio.gather(
        stage_fetch(all_headers, conn, classify_queue, executor, loop),
        stage_classify(classify_queue, delete_queue, conn),
        stage_delete(delete_queue, all_classified),
    )

    conn.close()
    executor.shutdown(wait=False)

    # -------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------

    ts      = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    summary = {
        "KEEP":   sum(1 for e in all_classified if e.get("category") == "KEEP"),
        "DELETE": sum(1 for e in all_classified if e.get("category") == "DELETE"),
        "REVIEW": sum(1 for e in all_classified if e.get("category") == "REVIEW"),
    }

    all_path = f"delete_email_classification_{ts}.json"
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump({
            "classified_at": datetime.datetime.now().isoformat(),
            "dry_run": DRY_RUN,
            "total": len(all_classified),
            "summary": summary,
            "emails": all_classified,
        }, f, indent=2, ensure_ascii=False)

    review_emails = [e for e in all_classified if e.get("category") == "REVIEW"]
    review_path   = f"review_email_classification_{ts}.json"
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump({"total": len(review_emails), "emails": review_emails},
                  f, indent=2, ensure_ascii=False)

    # -------------------------------------------------
    # FINAL REPORT
    # -------------------------------------------------

    end     = datetime.datetime.now()
    elapsed = str(end - start).split(".")[0]

    print("\n" + "=" * 60)
    print("📋 FINAL REPORT")
    print("=" * 60)
    print(f"  ✅ KEEP   : {summary['KEEP']}")
    print(f"  🗑️  DELETE : {summary['DELETE']}")
    print(f"  🔍 REVIEW : {summary['REVIEW']}")
    print(f"  📊 Total  : {len(all_classified)}")
    print(f"\n  💾 Full results  → {all_path}")
    print(f"  💾 Review emails → {review_path}")
    if DRY_RUN:
        print("\n  ⚠️  DRY RUN — nothing was deleted. "
              "Set DRY_RUN = False to enable deletion.")
    print(f"\n⏹️  Finished at: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total time : {elapsed}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_pipeline())