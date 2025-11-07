"""Worker orchestration utilities for planner-worker workflow."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from agentflow.models.memory import Memory
from agentflow.models.utils import make_json_serializable_truncated

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from agentflow.models.executor import Executor
    from agentflow.models.planner import Planner
    from agentflow.models.formatters import PlanStep, WorkerProgressCheck, WorkerSpecification, WorkerSummary
else:
    PlanStep = Any
    WorkerSpecification = Any
    WorkerProgressCheck = Any
    WorkerSummary = Any


@dataclass
class WorkerRunReport:
    """Structured result returned after executing a worker assignment."""

    worker_name: str
    plan_step_id: int
    status: str
    summary: str
    deliverables: List[str] = field(default_factory=list)
    follow_up: str = ""
    actions: List[Dict[str, Any]] = field(default_factory=list)
    progress_history: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_time: float = 0.0
    memory_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_name": self.worker_name,
            "plan_step_id": self.plan_step_id,
            "status": self.status,
            "summary": self.summary,
            "deliverables": self.deliverables,
            "follow_up": self.follow_up,
            "actions": self.actions,
            "progress_history": self.progress_history,
            "elapsed_time": self.elapsed_time,
            "memory_snapshot": self.memory_snapshot,
        }


class WorkerAgent:
    """Executable worker that collaborates with the planner to solve subtasks."""

    def __init__(
        self,
        planner: Planner,
        executor: Executor,
        specification: WorkerSpecification,
        verbose: bool = False,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.verbose = verbose
        self.specification = specification
        self.memory = Memory()

    def describe(self) -> Dict[str, Any]:
        """Return metadata describing the worker for planner prompts."""

        return {
            "worker_name": self.specification.worker_name,
            "worker_role": self.specification.worker_role,
            "mission": self.specification.mission,
            "tool_names": self.specification.tool_names,
            "success_criteria": self.specification.success_criteria,
        }

    def update_assignment(
        self,
        specification: WorkerSpecification,
        reset_memory: bool = True,
    ) -> None:
        """Update the worker specification, optionally clearing memory."""

        self.specification = specification
        if reset_memory:
            self.memory = Memory()

    def run(
        self,
        question: str,
        image: Optional[str],
        plan_step: PlanStep,
        global_memory: Memory,
        max_steps: int,
        max_time: int,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> WorkerRunReport:
        """Execute the assigned plan step using planner-guided actions."""

        start_time = time.time()
        actions: List[Dict[str, Any]] = []
        progress_history: List[Dict[str, Any]] = []
        last_progress: Optional[WorkerProgressCheck] = None

        for step_index in range(1, max_steps + 1):
            if time.time() - start_time >= max_time:
                if self.verbose:
                    print(
                        f"Worker {self.specification.worker_name} reached max_time {max_time}s at step {step_index}",
                    )
                break

            next_step = self.planner.generate_worker_step(
                question=question,
                image=image,
                plan_step=plan_step,
                worker_spec=self.specification,
                worker_memory=self.memory,
                global_memory=global_memory,
                step_count=step_index,
                max_step_count=max_steps,
                json_data=json_data,
            )
            context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(
                next_step,
                allowed_tools=self.specification.tool_names,
            )

            if tool_name is None:
                actions.append(
                    {
                        "step": step_index,
                        "tool_name": None,
                        "sub_goal": sub_goal,
                        "context": context,
                        "command": None,
                        "result": "Planner selected an unavailable tool. Worker stopped.",
                    },
                )
                break

            tool_metadata = self.planner.toolbox_metadata.get(tool_name, {})
            tool_command = self.executor.generate_tool_command(
                question,
                image,
                context or "",
                sub_goal or plan_step.objective,
                tool_name,
                tool_metadata,
                step_count=step_index,
                json_data=None,
            )
            analysis, explanation, command = self.executor.extract_explanation_and_command(tool_command)
            execution_result = self.executor.execute_tool_command(tool_name, command)
            execution_result = make_json_serializable_truncated(execution_result)

            self.memory.add_action(
                step_index,
                tool_name,
                sub_goal or plan_step.objective,
                command,
                execution_result,
            )
            actions.append(
                {
                    "step": step_index,
                    "tool_name": tool_name,
                    "sub_goal": sub_goal,
                    "context": context,
                    "analysis": analysis,
                    "explanation": explanation,
                    "command": command,
                    "result": execution_result,
                },
            )

            progress = self.planner.evaluate_worker_progress(
                question=question,
                plan_step=plan_step,
                worker_spec=self.specification,
                worker_memory=self.memory,
                global_memory=global_memory,
                json_data=json_data,
            )
            progress_history.append(progress.model_dump())
            last_progress = progress

            if progress.status.lower().startswith("complete"):
                break

        summary = self.planner.summarize_worker_result(
            question=question,
            plan_step=plan_step,
            worker_spec=self.specification,
            worker_memory=self.memory,
            global_memory=global_memory,
            json_data=json_data,
        )

        elapsed_time = round(time.time() - start_time, 2)
        status = last_progress.status if last_progress else "continue"

        return WorkerRunReport(
            worker_name=self.specification.worker_name,
            plan_step_id=plan_step.step_id,
            status=status,
            summary=summary.summary,
            deliverables=summary.deliverables,
            follow_up=summary.follow_up,
            actions=actions,
            progress_history=progress_history,
            elapsed_time=elapsed_time,
            memory_snapshot=self.memory.get_actions(),
        )


class WorkerManager:
    """Manage lifecycle of workers within the planner-worker workflow."""

    def __init__(
        self,
        planner: Planner,
        executor: Executor,
        verbose: bool = False,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.verbose = verbose
        self._workers: Dict[str, WorkerAgent] = {}

    def list_workers(self) -> List[Dict[str, Any]]:
        return [worker.describe() for worker in self._workers.values()]

    def get_or_create_worker(
        self,
        specification: WorkerSpecification,
        reset_memory: bool = True,
    ) -> WorkerAgent:
        worker = self._workers.get(specification.worker_name)
        if worker is None:
            worker = WorkerAgent(
                planner=self.planner,
                executor=self.executor,
                specification=specification,
                verbose=self.verbose,
            )
            self._workers[specification.worker_name] = worker
        else:
            worker.update_assignment(specification, reset_memory=reset_memory)
        return worker