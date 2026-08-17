from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# github pipeline logs retrieval
import os
import io
import zipfile
import requests
from github import Github, Auth


def get_failed_pipeline_logs():
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
    g = Github(auth=auth)
    repo = g.get_repo("zero101010/gs-rest-service")

    # Get latest workflow run
    workflow = repo.get_workflow("deploy.yml")
    runs = workflow.get_runs()

    failed_runs = [run for run in runs if run.conclusion == "failure"]
    failed_one = failed_runs[0]

    print(f"Run ID: {failed_one.id}, Conclusion: {failed_one.conclusion}")

    # Step 1: get the S3 redirect URL
    redirect = requests.get(
        failed_one.logs_url,
        headers={"Authorization": f"token {os.getenv('GITHUB_TOKEN')}"},
        allow_redirects=False,
    )
    download_url = redirect.headers["Location"]

    # Step 2: download logs zip 
    response = requests.get(download_url)

    # Step 3: extract and print failed step logs
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        for name in z.namelist():
            content = z.read(name).decode("utf-8", errors="replace")
            # Only print files that contain errors
            if "error" in content.lower() or "failed" in content.lower():
                print(f"\n--- {name} ---")
                print(content[len(content) - 8127 :])
                return content[len(content) - 8127 :]
    



@tool
def get_logs() -> str:
    """Retrieve the logs of the latest failed pipeline run."""
    return get_failed_pipeline_logs() or "No failed logs found."

@tool
def fix_pipeline(logs: str) -> str:
    """Suggest possible fixes for the pipeline given its error logs."""
    return f"Analyze these logs and suggest a fix:\n\n{logs}"

llm = ChatOpenAI(
    model="Qwen/Qwen3-8B",
    base_url="https://wvbb6j4jfr20uj-8000.proxy.runpod.net/v1",
    api_key="sk-wvbb6j4jfr20uj",
)

agent = create_react_agent(
    model=llm,
    tools=[get_logs, fix_pipeline],
    prompt="You are a helpful assistant that fixes failed CI/CD pipelines. First call get_logs to retrieve the error logs, then pass those logs to fix_pipeline to suggest a fix.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "What's the issue with the latest failed pipeline and how can we fix it?"}]})
# write file with the output
with open("pipeline_fix_suggestion.txt", "w") as f:
    f.write(result["messages"][-1].content)
print(result["messages"][-1].content)
