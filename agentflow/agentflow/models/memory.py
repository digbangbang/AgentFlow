from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ActionRecord:
    """Represents a single tool interaction."""

    step: int
    tool_name: str
    sub_goal: str
    command: str
    result: Any

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


class WorkerMemory:
    """Execution memory scoped to a single worker agent."""

    def __init__(self) -> None:
        self.actions: List[ActionRecord] = []
        self.notes: List[str] = []

    def add_action(
        self,
        step: int,
        tool_name: str,
        sub_goal: str,
        command: str,
        result: Any,
    ) -> None:
        self.actions.append(
            ActionRecord(
                step=step,
                tool_name=tool_name,
                sub_goal=sub_goal,
                command=command,
                result=result,
            ),
        )

    def add_note(self, note: str) -> None:
        self.notes.append(note)

    def get_actions(self) -> List[Dict[str, Any]]:
        return [record.to_dict() for record in self.actions]

    def clear(self) -> None:
        self.actions.clear()
        self.notes.clear()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "actions": self.get_actions(),
            "notes": list(self.notes),
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        self.clear()
        for record in snapshot.get("actions", []):
            self.actions.append(
                ActionRecord(
                    step=record.get("step", len(self.actions) + 1),
                    tool_name=record.get("tool_name", ""),
                    sub_goal=record.get("sub_goal", ""),
                    command=record.get("command", ""),
                    result=record.get("result"),
                ),
            )
        self.notes.extend(snapshot.get("notes", []))


class PlannerMemory:
    """Planner-level working memory and structured state store."""

    def __init__(self) -> None:
        self.user_inputs: List[str] = []
        self.system_notes: List[str] = []
        self.reasoning_traces: List[str] = []
        self.plan_history: List[Dict[str, Any]] = []
        self.worker_specs: Dict[int, Dict[str, Any]] = {}
        self.worker_steps: Dict[int, List[Dict[str, Any]]] = {}
        self.worker_progress: Dict[int, List[Dict[str, Any]]] = {}
        self.worker_reports: Dict[int, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {}

    def add_user_input(self, content: str) -> None:
        if content:
            self.user_inputs.append(content)

    def add_system_note(self, note: str) -> None:
        if note:
            self.system_notes.append(note)

    def add_reasoning_trace(self, trace: str) -> None:
        if trace:
            self.reasoning_traces.append(trace)

    def record_plan(self, plan_outline: Any) -> None:
        if plan_outline is None:
            return
        payload = (
            plan_outline.model_dump()
            if hasattr(plan_outline, "model_dump")
            else plan_outline
        )
        self.plan_history.append(payload)

    def record_worker_spec(self, plan_step_id: int, spec: Any) -> None:
        payload = spec.model_dump() if hasattr(spec, "model_dump") else spec
        self.worker_specs[plan_step_id] = payload

    def record_worker_step(self, plan_step_id: int, step_payload: Any) -> None:
        payload = (
            step_payload.model_dump()
            if hasattr(step_payload, "model_dump")
            else step_payload
        )
        self.worker_steps.setdefault(plan_step_id, []).append(payload)

    def record_worker_progress(self, plan_step_id: int, progress: Any) -> None:
        payload = (
            progress.model_dump()
            if hasattr(progress, "model_dump")
            else progress
        )
        self.worker_progress.setdefault(plan_step_id, []).append(payload)

    def record_worker_report(self, plan_step_id: int, report: Any) -> None:
        payload = (
            report.model_dump()
            if hasattr(report, "model_dump")
            else report
        )
        self.worker_reports[plan_step_id] = payload

    def snapshot(self) -> Dict[str, Any]:
        return {
            "user_inputs": list(self.user_inputs),
            "system_notes": list(self.system_notes),
            "reasoning_traces": list(self.reasoning_traces),
            "plan_history": list(self.plan_history),
            "worker_specs": dict(self.worker_specs),
            "worker_steps": {k: list(v) for k, v in self.worker_steps.items()},
            "worker_progress": {k: list(v) for k, v in self.worker_progress.items()},
            "worker_reports": dict(self.worker_reports),
            "metadata": dict(self.metadata),
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        self.user_inputs = list(snapshot.get("user_inputs", []))
        self.system_notes = list(snapshot.get("system_notes", []))
        self.reasoning_traces = list(snapshot.get("reasoning_traces", []))
        self.plan_history = list(snapshot.get("plan_history", []))
        self.worker_specs = dict(snapshot.get("worker_specs", {}))
        self.worker_steps = {
            int(k): list(v)
            for k, v in snapshot.get("worker_steps", {}).items()
        }
        self.worker_progress = {
            int(k): list(v)
            for k, v in snapshot.get("worker_progress", {}).items()
        }
        self.worker_reports = {
            int(k): v
            for k, v in snapshot.get("worker_reports", {}).items()
        }
        self.metadata = dict(snapshot.get("metadata", {}))


class GlobalMemory:
    """Conversation-level memory for the entire solving session."""

    def __init__(self) -> None:
        self.query: Optional[str] = None
        self.files: List[Dict[str, str]] = []
        self.actions: Dict[str, ActionRecord] = {}
        self._init_file_types()

    def set_query(self, query: str) -> None:
        if not isinstance(query, str):
            raise TypeError("Query must be a string")
        self.query = query

    def _init_file_types(self) -> None:
        self.file_types = {
            "image": [".jpg", ".jpeg", ".png", ".gif", ".bmp"],
            "text": [".txt", ".md"],
            "document": [".pdf", ".doc", ".docx"],
            "code": [".py", ".js", ".java", ".cpp", ".h"],
            "data": [".json", ".csv", ".xml"],
            "spreadsheet": [".xlsx", ".xls"],
            "presentation": [".ppt", ".pptx"],
        }
        self.file_type_descriptions = {
            "image": "An image file ({ext} format) provided as context for the query",
            "text": "A text file ({ext} format) containing additional information related to the query",
            "document": "A document ({ext} format) with content relevant to the query",
            "code": "A source code file ({ext} format) potentially related to the query",
            "data": "A data file ({ext} format) containing structured data pertinent to the query",
            "spreadsheet": "A spreadsheet file ({ext} format) with tabular data relevant to the query",
            "presentation": "A presentation file ({ext} format) with slides related to the query",
        }

    def _get_default_description(self, file_name: str) -> str:
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()

        for file_type, extensions in self.file_types.items():
            if ext in extensions:
                return self.file_type_descriptions[file_type].format(ext=ext[1:])

        return f"A file with {ext[1:]} extension, provided as context for the query"

    def add_file(
        self,
        file_name: Union[str, List[str]],
        description: Union[str, List[str], None] = None,
    ) -> None:
        if isinstance(file_name, str):
            file_name = [file_name]

        if description is None:
            description = [self._get_default_description(fname) for fname in file_name]
        elif isinstance(description, str):
            description = [description]

        if len(file_name) != len(description):
            raise ValueError("The number of files and descriptions must match.")

        for fname, desc in zip(file_name, description):
            self.files.append({
                "file_name": fname,
                "description": desc,
            })

    def add_action(self, step_count: int, tool_name: str, sub_goal: str, command: str, result: Any) -> None:
        record = ActionRecord(
            step=step_count,
            tool_name=tool_name,
            sub_goal=sub_goal,
            command=command,
            result=result,
        )
        step_name = f"Action Step {step_count}"
        self.actions[step_name] = record

    def get_query(self) -> Optional[str]:
        return self.query

    def get_files(self) -> List[Dict[str, str]]:
        return list(self.files)

    def get_actions(self) -> Dict[str, Dict[str, Any]]:
        return {key: record.to_dict() for key, record in self.actions.items()}

    def snapshot(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "files": list(self.files),
            "actions": self.get_actions(),
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        self.query = snapshot.get("query")
        self.files = list(snapshot.get("files", []))
        self.actions = {}
        for step_name, payload in snapshot.get("actions", {}).items():
            self.actions[step_name] = ActionRecord(
                step=payload.get("step", len(self.actions) + 1),
                tool_name=payload.get("tool_name", ""),
                sub_goal=payload.get("sub_goal", ""),
                command=payload.get("command", ""),
                result=payload.get("result"),
            )


# Backwards compatibility export
Memory = GlobalMemory
    