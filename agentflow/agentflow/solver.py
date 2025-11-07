import argparse
import time
import json
from typing import Optional

from agentflow.models.initializer import Initializer
from agentflow.models.planner import Planner
from agentflow.models.memory import Memory
from agentflow.models.executor import Executor
from agentflow.models.worker import WorkerManager

class Solver:
    def __init__(
        self,
        planner: Planner,
        memory: Memory,
        executor: Executor,
        output_types: str = "base,final,direct",
        max_steps: int = 10,
        max_subtasks: int = 5,
        max_time: int = 300,
        max_tokens: int = 4000,
        root_cache_dir: str = "cache",
        verbose: bool = True, 
        temperature: float = .0
    ):
        self.planner = planner
        self.memory = memory
        self.executor = executor
        self.max_steps = max_steps
        self.max_subtasks = max_subtasks
        self.max_time = max_time
        self.max_tokens = max_tokens
        self.root_cache_dir = root_cache_dir

        self.output_types = output_types.lower().split(',')
        self.temperature  = temperature
        assert all(output_type in ["base", "final", "direct"] for output_type in self.output_types), "Invalid output type. Supported types are 'base', 'final', 'direct'."
        self.verbose = verbose

    def solve(self, question: str, image_path: Optional[str] = None):
        """Solve a query using the planner-worker workflow."""

        self.executor.set_query_cache_dir(self.root_cache_dir)
        self.memory.set_query(question)

        json_data = {
            "query": question,
            "image": image_path,
        }

        if self.verbose:
            print(f"\n==> 🔍 Received Query: {question}")
            if image_path:
                print(f"\n==> 🖼️ Received Image: {image_path}")

        if "base" in self.output_types:
            base_response = self.planner.generate_base_response(
                question,
                image_path,
                self.max_tokens,
            )
            json_data["base_response"] = base_response
            if self.verbose:
                print(f"\n==> 📝 Base Response from LLM:\n\n{base_response}")

        if set(self.output_types) == {"base"}:
            return json_data

        start_time = time.time()

        plan_outline = self.planner.generate_plan(
            question=question,
            image=image_path,
            max_subtasks=self.max_subtasks,
            json_data=json_data,
        )
        json_data["plan"] = plan_outline.model_dump()

        if self.verbose:
            print("\n==> 🧭 Planner Roadmap")
            print(plan_outline.reasoning)
            for step in plan_outline.steps:
                tool_list = ", ".join(step.suggested_tools) if step.suggested_tools else "(planner to decide)"
                print(
                    f"  - Step {step.step_id}: {step.title}\n"
                    f"    Objective: {step.objective}\n"
                    f"    Tools: {tool_list}\n"
                    f"    Success Criteria: {step.success_criteria}\n",
                )

        worker_manager = WorkerManager(
            planner=self.planner,
            executor=self.executor,
            verbose=self.verbose,
        )

        worker_reports = []
        global_step_counter = 0

        for plan_step in plan_outline.steps:
            if self.verbose:
                print(f"\n==> 🗂️ Planning Worker for Step {plan_step.step_id}: {plan_step.title}")

            worker_spec = self.planner.design_worker(
                question=question,
                plan_step=plan_step,
                global_memory=self.memory,
                existing_workers=worker_manager.list_workers(),
                image=image_path,
                json_data=json_data,
            )

            if self.verbose:
                print(
                    f"    Worker: {worker_spec.worker_name} ({worker_spec.worker_role})\n"
                    f"    Mission: {worker_spec.mission}\n"
                    f"    Tools: {', '.join(worker_spec.tool_names) if worker_spec.tool_names else 'None'}",
                )

            worker = worker_manager.get_or_create_worker(worker_spec) # TODO
            worker_report = worker.run(
                question=question,
                image=image_path,
                plan_step=plan_step,
                global_memory=self.memory,
                max_steps=self.max_steps,
                max_time=self.max_time,
                json_data=json_data,
            )

            worker_report_dict = worker_report.to_dict()
            worker_reports.append(worker_report_dict)

            for action in worker_report_dict["actions"]:
                global_step_counter += 1
                tool_name = action.get("tool_name") or "Unavailable_Tool"
                sub_goal = action.get("sub_goal") or plan_step.objective
                command = action.get("command") or ""
                result = action.get("result")
                self.memory.add_action(
                    global_step_counter,
                    tool_name,
                    sub_goal,
                    command,
                    result,
                )

            if self.verbose:
                print(
                    f"    ✅ Worker {worker_spec.worker_name} status: {worker_report_dict['status']}\n"
                    f"    Summary: {worker_report_dict['summary']}\n",
                )

        json_data["workers"] = worker_reports
        json_data["memory"] = self.memory.get_actions()
        json_data["step_count"] = global_step_counter
        json_data["execution_time"] = round(time.time() - start_time, 2)
        json_data["plan_completed"] = all(
            report.get("status", "").lower().startswith("complete")
            for report in worker_reports
            if report
        )

        if "final" in self.output_types:
            final_output = self.planner.generate_final_output(
                question,
                image_path,
                self.memory,
            )
            json_data["final_output"] = final_output
            if self.verbose:
                print(f"\n==> 🐙 Detailed Solution:\n\n{final_output}")

        if "direct" in self.output_types:
            direct_output = self.planner.generate_direct_output(
                question,
                image_path,
                self.memory,
            )
            json_data["direct_output"] = direct_output
            if self.verbose:
                print(f"\n==> 🐙 Final Answer:\n\n{direct_output}")

        if self.verbose:
            print(f"\n[Total Time]: {json_data['execution_time']}s")
            print("\n==> ✅ Query Solved!")

        return json_data

def construct_solver(llm_engine_name : str = "gpt-4o",
                     enabled_tools : list[str] = ["all"],
                     tool_engine: list[str] = ["Default"],
                     output_types : str = "final,direct",
                     max_steps : int = 10,
                     max_time : int = 300,
                     max_tokens : int = 4000,
                     root_cache_dir : str = "solver_cache",
                     verbose : bool = True,
                     vllm_config_path : str = None,
                     base_url : str = None,
                     temperature: float = 0.0
                     ):
    
    # Instantiate Initializer
    initializer = Initializer(
        enabled_tools=enabled_tools,
        tool_engine=tool_engine,
        model_string=llm_engine_name,
        verbose=verbose,
        vllm_config_path=vllm_config_path,
    )

    # Instantiate Planner
    planner = Planner(
        llm_engine_name=llm_engine_name,
        toolbox_metadata=initializer.toolbox_metadata,
        available_tools=initializer.available_tools,
        verbose=verbose,
        base_url=base_url,
        temperature=temperature
    )

    # Instantiate Memory
    memory = Memory()

    # Instantiate Executor with tool instances cache
    executor = Executor(
        llm_engine_name=llm_engine_name,
        # llm_engine_name="dashscope",
        root_cache_dir=root_cache_dir,
        verbose=verbose,
        base_url=base_url,
        temperature=temperature,
        tool_instances_cache=initializer.tool_instances_cache  # Pass the cached tool instances
    )

    # Instantiate Solver
    solver = Solver(
        planner=planner,
        memory=memory,
        executor=executor,
        output_types=output_types,
        max_steps=max_steps,
        max_time=max_time,
        max_tokens=max_tokens,
        root_cache_dir=root_cache_dir,
        verbose=verbose,
        temperature=temperature
    )
    return solver

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the agentflow demo with specified parameters.")
    parser.add_argument("--llm_engine_name", default="vllm-Qwen/Qwen3-8B", help="LLM engine name.")
    parser.add_argument(
        "--output_types",
        default="base,final,direct",
        help="Comma-separated list of required outputs (base,final,direct)"
    )
    parser.add_argument("--enabled_tools", default="Base_Generator_Tool", help="List of enabled tools.")
    parser.add_argument("--root_cache_dir", default="solver_cache", help="Path to solver cache directory.")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Maximum tokens for LLM generation.")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of steps to execute.")
    parser.add_argument("--max_time", type=int, default=300, help="Maximum time allowed in seconds.")
    parser.add_argument("--verbose", type=bool, default=True, help="Enable verbose output.")
    return parser.parse_args()
    
def main(args):
    tool_engine=["self","self",]
    solver = construct_solver(
        llm_engine_name=args.llm_engine_name,
        enabled_tools=["Base_Generator_Tool","Python_Coder_Tool"],
        tool_engine=tool_engine,
        output_types=args.output_types,
        max_steps=args.max_steps,
        max_time=args.max_time,
        max_tokens=args.max_tokens,
        base_url="http://localhost:8080/v1",
        verbose=args.verbose,
        temperature=0.7
    )

    # Solve the task or problem
    solver.solve("What is the capital of France?")
    # solver.solve(
    #     "Below is a piece of text. Please count the top 5 most frequent English words and provide a brief summary of the themes expressed by these high-frequency words: Text:'Machine learning enables computers to learn from data. Machine learning models improve as more data becomes available. Data-driven methods are essential in modern machine learning workflows.'"
    # )
    # solver.solve(
    #     "Here are three numbers: 3, 7, 2. Please calculate their average and tell me in one sentence what this average indicates."
    # )

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
