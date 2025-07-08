import os
import json
import subprocess
import requests
import logging
from icecream import ic
from datetime import datetime
from multi_agents import MultiLangChainAgents

TASK_API_URL = "http://localhost:8081/task/index/"  # API endpoint for SWE-Bench-Lite
TEST_API_URL = "http://localhost:8084/test"  # API endpoint for SWE-Bench-Lite tests

REPOS_DIR = os.path.abspath("./repos") # Set this to the path of your repository if needed
LOGS_DIR = os.path.abspath("./logs")  # Set this to the path of your logs if needed
MAX_FEEDBACK = 8

os.makedirs(REPOS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(LOGS_DIR, 'main_langchain.log'))
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


def handle_task(index):
    logger.info(f"\n___ TASK {index} ___")
    
    task_api_url = f"{TASK_API_URL}{index}"
    repo_dir = os.path.join(REPOS_DIR, f"repo_{index}")
    start_dir = os.getcwd() 
    
    logger.info(f" - Repository {repo_dir} - ")
    
    # Fetch the task details from the API
    response = requests.get(task_api_url)
    if response.status_code != 200:
        raise Exception(f"Invalid Task API response: {response.status_code}")
    testcase = response.json()
    prompt = testcase["Problem_statement"]
    git_clone = testcase["git_clone"]
    fail_tests = json.loads(testcase.get("FAIL_TO_PASS", "[]"))
    pass_tests = json.loads(testcase.get("PASS_TO_PASS", "[]"))
    instance_id = testcase["instance_id"]
    parts = git_clone.split("&&")
    clone_part = parts[0].strip()
    checkout_part = parts[-1].strip() if len(parts) > 1 else None
    repo_url = clone_part.split()[2]
    commit_hash = checkout_part.split()[-1] if checkout_part else "main"
    
    # set up the repo directory
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True, env=env)
    else:
        ic(f"Repo {repo_dir} already exists – skipping clone.")

    os.chdir(repo_dir)
    subprocess.run(["git", "checkout", commit_hash], cwd=repo_dir, check=True, env=env)
    
    # ___________________________________________
    # Run the Multi LangChain Agents
    try:
        if repo_dir and prompt: 
            logger.info(f"Repository: {repo_dir}, using {len(prompt)} length prompt")
        multi_agents = MultiLangChainAgents(repo_path=repo_dir, log_dir=LOGS_DIR)
        multi_agents.run(task=prompt, max_feedback=MAX_FEEDBACK)
        
        try:
            logger.info(f"Total Costs: ${multi_agents.get_total_cost():.2f}")
            logger.info(f"Total Tokens: {multi_agents.get_total_tokens()}")
        except Exception as e:
            logger.error(f"Error occurred while getting usage metrics: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while running langchain agents in task: {e}")
    
    try:
        env = os.environ.copy()
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, env=env)
        ic("Git add completed.")
        # subprocess.run(["git", "commit", "-m", f"Task {index} solved by LangChain Agents"], cwd=repo_dir, check=True, env=env)
        # ic("Git commit completed.")
    except subprocess.CalledProcessError as e:
        print(f"Git commit failed: {e}")
    
    # __________________________________________
    # Run tests
    test_payload = {
        "instance_id": instance_id,
        "repoDir": f"/repos/repo_{index}",  # mount with docker
        "FAIL_TO_PASS": fail_tests,
        "PASS_TO_PASS": pass_tests
    }
    res = None
    try:
        res = requests.post(TEST_API_URL, json=test_payload)
        try:
            ic(res.status_code)
            ic(res.text)
            logger.info(f"Test API response: {res.status_code} - {res.text}")
            res.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Error occurred while running tests: {e}")
            
        result_raw = res.json().get("harnessOutput", "{}")
        result_json = json.loads(result_raw)
        if not result_json:
            raise ValueError("No data in harnessOutput – possible evaluation error or empty result")
        instance_id = next(iter(result_json))
        tests_status = result_json[instance_id]["tests_status"]
        fail_pass_results = tests_status["FAIL_TO_PASS"]
        fail_pass_total = len(fail_pass_results["success"]) + len(fail_pass_results["failure"])
        fail_pass_passed = len(fail_pass_results["success"])
        pass_pass_results = tests_status["PASS_TO_PASS"]
        pass_pass_total = len(pass_pass_results["success"]) + len(pass_pass_results["failure"])
        pass_pass_passed = len(pass_pass_results["success"])
        os.chdir(start_dir)
        logger.info(f"FAIL_TO_PASS passed: {fail_pass_passed}/{fail_pass_total}\n")
        logger.info(f"PASS_TO_PASS passed: {pass_pass_passed}/{pass_pass_total}\n")
    except requests.RequestException as e:
        os.chdir(start_dir)
        logger.error(f"An error occurred while running tests: {e}")
        logger.error(f"Response content: {res if res else 'No response'}")
    
    # reset to start directory
    os.chdir(start_dir)   

    

if __name__ == "__main__":
    logger.info(f"Starting Multi LangChain Agents at {datetime.now().isoformat()}\n")
    
    for i in range(3, 31):
        try:
            handle_task(i)
        except Exception as e:
            print(f"An error occurred while handling task {i}: {e}")
            logger.error(f"An error occurred while handling task {i}: {e}")
