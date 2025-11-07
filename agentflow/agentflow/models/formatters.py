from typing import List

from pydantic import BaseModel, Field

# Planner: QueryAnalysis
class QueryAnalysis(BaseModel):
    concise_summary: str
    required_skills: str
    relevant_tools: str
    additional_considerations: str

    def __str__(self):
        return f"""
Concise Summary: {self.concise_summary}

Required Skills:
{self.required_skills}

Relevant Tools:
{self.relevant_tools}

Additional Considerations:
{self.additional_considerations}
"""

# Planner: NextStep
class NextStep(BaseModel):
    justification: str
    context: str
    sub_goal: str
    tool_name: str

# Executor: MemoryVerification
class MemoryVerification(BaseModel):
    analysis: str
    stop_signal: bool

# Executor: ToolCommand
class ToolCommand(BaseModel):
    analysis: str
    explanation: str
    command: str


# Planner-Worker workflow: plan decomposition
class PlanStep(BaseModel):
    step_id: int
    title: str
    objective: str
    success_criteria: str
    suggested_tools: List[str] = Field(default_factory=list)
    handoff_notes: str = ""


class PlanOutline(BaseModel):
    reasoning: str
    steps: List[PlanStep]


class WorkerSpecification(BaseModel):
    worker_name: str
    worker_role: str
    mission: str
    context: str
    tool_names: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    system_prompt: str


class WorkerProgressCheck(BaseModel):
    analysis: str
    status: str
    blockers: str
    next_action: str


class WorkerSummary(BaseModel):
    summary: str
    follow_up: str
