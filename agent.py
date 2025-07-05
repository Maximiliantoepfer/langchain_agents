from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain_community.tools.file_management import (
    WriteFileTool,
    ReadFileTool,
    ListDirectoryTool,
)
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from config import MODEL, OPENAI_API_KEY, OPENAI_API_BASE


class LangChainAgent:
    def __init__(self, name: str, role: str, description: str, root_dir: str):
        self.name = name
        self.role = role
        self.description = description
        self.root_dir = root_dir

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        self.total_cost = 0.0
        self.total_tokens = 0

        self.tools = [
            WriteFileTool(root_dir=self.root_dir),
            ReadFileTool(root_dir=self.root_dir),
            ListDirectoryTool(root_dir=self.root_dir),
        ]

        self.llm = ChatOpenAI(
            model_name=MODEL,
            base_url=OPENAI_API_BASE,
            api_key=OPENAI_API_KEY,
        )

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            memory=self.memory,
        )
        

    def run(self, input_text: str) -> str:
        prompt = (
            f"<AGENT_ROLE>{self.role}</AGENT_ROLE>\n"
            f"<AGENT_DESCRIPTION>{self.description}</AGENT_DESCRIPTION>\n"
            f"<TOOL_FORMATTING>Use the write_file tool with arguments: `file_path` for the filename and `text` for the full file content (not content).</TOOL_FORMATTING>"
            f"<TASK>{input_text}</TASK>"
        )
        result = ""
        with get_openai_callback() as cb:
            result = self.agent.run(prompt)
            self.total_cost += cb.total_cost
            self.total_tokens += cb.total_tokens
        return result
    
    def get_total_cost(self):
        return self.total_cost
    
    def get_total_tokens(self):
        return self.total_tokens