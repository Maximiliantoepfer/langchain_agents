import os
import logging
from agent import LangChainAgent

class MultiLangChainAgents():
    def __init__(self, repo_path: str = '', log_dir: str = './'):
        self.repo_path = repo_path
        self.logger = logging.getLogger(__name__)
        filehandler = logging.FileHandler(os.path.join(log_dir, 'multi-langchain-agents.log'))
        filehandler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(filehandler)
        self.logger.setLevel(logging.INFO)

        # Create agents
        self.planner = LangChainAgent(
            name="Planner",
            role="Architect and Planning Specialist",
            description=(
                "You're responsible for analyzing the given problem and determining which files need to be created, modified, or deleted."
                " Your output must be a concrete plan that lists file paths, summarizes the current content of these files if relevant,"
                " and clearly outlines what changes should be made and why."
                " You should revise your plan when feedback indicates incorrect or incomplete changes."
            ),
            root_dir=self.repo_path,
        )

        self.coder = LangChainAgent(
            name="Coder",
            role="Senior Python Developer",
            description=(
                "You are provided with a file-specific plan from the Planner."
                " Based on this plan, make the necessary modifications in the codebase."
                " You should read, edit, and write files using the repository tools.  If a file must change, regenerate it fully with your changes."
                " Never decide what files to touch — only follow the Planner's file list and plan."
            ),
            root_dir=self.repo_path,
        )

        self.tester = LangChainAgent(
            name="Tester",
            role="QA and Code Reviewer",
            description=(
                "You evaluate the changes made by the Coder. Run tests or review logic depending on the situation."
                " You must give concise feedback identifying broken logic or implementation flaws."
                " If no problems are found, respond with the word TERMINATE."
                " If Planner's assumptions are incorrect, suggest revisiting the plan."
            ),
            root_dir=self.repo_path,
        )

    def run(self, task: str, max_feedback: int = 5):
        self.logger.info(f"Starting run for task: {task}")
        
        # Step 1: Initial plan
        plan = self.planner.run(task)
        self.logger.info("[PLANNER OUTPUT]")
        self.logger.info(plan)

        feedback_round = 0
        max_feedback = max_feedback

        while feedback_round < max_feedback:
            feedback_round += 1
            self.logger.info(f"--- FEEDBACK ROUND {feedback_round} ---")

            # Step 2: Coder implementation plan
            coder_response = self.coder.run(plan)
            self.logger.info("[CODER RESPONSE]")
            self.logger.info(coder_response)

            # Step 3: Tester gives feedback
            feedback = self.tester.run(coder_response)
            self.logger.info("[TESTER FEEDBACK]")
            self.logger.info(feedback)

            if "TERMINATE" in feedback.upper():
                self.logger.info("✅ TERMINATE signal received. Process complete.")
                break

            if "REPLAN" in feedback.upper():
                self.logger.info("🔁 REPLAN signal received. Tester requested a new plan. Planner will reanalyze the repository.")
                plan = self.planner.run(feedback)
                continue  # Go to next round with updated plan

            # Otherwise: coder fixes based on feedback (without replanning)
            plan = feedback  # let coder treat feedback like delta instructions

        else:
            self.logger.info("Max feedback rounds reached. Process stopped.")

    def get_total_cost(self):
        return self.planner.get_total_cost() + self.coder.get_total_cost() + self.tester.get_total_cost()
    
    def get_total_tokens(self):
        return self.planner.get_total_tokens() + self.coder.get_total_tokens() + self.tester.get_total_tokens()