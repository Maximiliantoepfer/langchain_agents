from icecream import ic
from multi_agents import MultiLangChainAgents



if __name__ == "__main__":
    # Working directory (shared by all agents)
    repo_path = "C:/Users/maxto/source/langchain/repos"
    problem = (
        "The repository contains a bug or missing implementation."
        " A file named 'greeting.py' should be created or modified to output the actual price of the MSCIWorld Index to the console."
    )
    multi_agents = MultiLangChainAgents(repo_path=repo_path)
    multi_agents.run(task=problem)

    print(multi_agents.get_total_cost())
    print(multi_agents.get_total_tokens())
