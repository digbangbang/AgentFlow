import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from agentflow.engine.factory import create_llm_engine
from agentflow.models.formatters import (
    MemoryVerification,
    NextStep,
    PlanOutline,
    PlanStep,
    QueryAnalysis,
    WorkerProgressCheck,
    WorkerSpecification,
    WorkerSummary,
)
from agentflow.models.memory import Memory


class Planner:
    def __init__(self, llm_engine_name: str, toolbox_metadata: dict = None, available_tools: List = None, 
    verbose: bool = False, base_url: str = None, is_multimodal: bool = False, check_model: bool = True, temperature : float = .0):
        self.llm_engine_name = llm_engine_name
        self.is_multimodal = is_multimodal
        # self.llm_engine_mm = create_llm_engine(model_string=llm_engine_name, is_multimodal=False, base_url=base_url, temperature = temperature)
        # self.llm_engine_fixed = create_llm_engine(model_string="dashscope", is_multimodal=False, temperature = temperature)
        self.llm_engine_fixed = create_llm_engine(model_string=llm_engine_name, is_multimodal=False, base_url=base_url, temperature = temperature)
        self.llm_engine = create_llm_engine(model_string=llm_engine_name, is_multimodal=False, base_url=base_url, temperature = temperature)
        self.toolbox_metadata = toolbox_metadata if toolbox_metadata is not None else {}
        self.available_tools = available_tools if available_tools is not None else []

        self.verbose = verbose

    def _coerce_plan_outline(self, raw_plan: Any, max_subtasks: int) -> PlanOutline:
        """Convert model output into PlanOutline, falling back if parsing fails."""
        parsed_payload = None

        txt = raw_plan.strip()
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
        
        try:
            parsed_payload = json.loads(txt)
        except Exception as exc:
            if self.verbose:
                print(f"Planner generate_plan: JSON parsing failed: {exc}")

        if parsed_payload is not None:
            try:
                plan = PlanOutline(**parsed_payload)
                return self._truncate_plan_steps(plan, max_subtasks)
            except Exception as exc:
                if self.verbose:
                    print(f"Planner generate_plan: PlanOutline construction failed: {exc}. Using fallback plan outline.")

        return self._build_fallback_plan(max_subtasks)

    def _truncate_plan_steps(self, plan: PlanOutline, max_subtasks: int) -> PlanOutline:
        if not plan.steps:
            return self._build_fallback_plan(max_subtasks)

        trimmed_steps = plan.steps[:max_subtasks]
        normalized_steps: List[PlanStep] = []
        for index, step in enumerate(trimmed_steps, start=1):
            suggested = list(step.suggested_tools) if step.suggested_tools else []
            normalized_steps.append(
                PlanStep(
                    step_id=index,
                    title=step.title,
                    objective=step.objective,
                    success_criteria=step.success_criteria,
                    suggested_tools=suggested,
                    handoff_notes=step.handoff_notes,
                ),
            )
        return PlanOutline(reasoning=plan.reasoning, steps=normalized_steps)

    def _build_fallback_plan(self, max_subtasks: int) -> PlanOutline:

        step_templates = [
            {
                "title": "Understand the query",
                "objective": "Summarize the task requirements and note available context",
                "success_criteria": "Key objectives and constraints captured clearly",
                "handoff_notes": "Provide a concise brief for execution",
            },
            {
                "title": "Execute tooling workflow",
                "objective": "Use the most relevant tools to address the core objective",
                "success_criteria": "Question answered or artefact produced with tool outputs",
                "handoff_notes": "Highlight important results for synthesis",
            },
            {
                "title": "Synthesize final response",
                "objective": "Compile findings into a coherent final answer",
                "success_criteria": "Final answer ready for user consumption",
                "handoff_notes": "",
            },
        ]

        steps_to_create = min(max_subtasks, len(step_templates)) or 1

        fallback_steps: List[PlanStep] = []
        for idx in range(steps_to_create):
            template = step_templates[idx]
            fallback_steps.append(
                PlanStep(
                    step_id=idx + 1,
                    title=template["title"],
                    objective=template["objective"],
                    success_criteria=template["success_criteria"],
                    suggested_tools=self.available_tools,
                    handoff_notes=template["handoff_notes"],
                ),
            )

        return PlanOutline(
            reasoning="Fallback plan generated because model output could not be parsed.",
            steps=fallback_steps,
        )

    def _normalize_tool_name(
        self,
        tool_name: str,
        allowed_tools: Optional[List[str]] = None,
    ) -> str:
        """Normalize tool names to match available tool identifiers."""

        allowed = allowed_tools if allowed_tools is not None else self.available_tools

        def to_canonical(name: str) -> str:
            return "_".join(part.lower() for part in re.split(r"[ _]+", name.strip()))

        normalized_input = to_canonical(tool_name)
        for tool in allowed:
            if to_canonical(tool) == normalized_input:
                return tool
        return tool_name

    def _filter_tool_metadata(self, tools: List[str]) -> Dict[str, Any]:
        """Return toolbox metadata limited to the provided tool names."""

        return {
            tool: self.toolbox_metadata.get(tool, {})
            for tool in tools
            if tool in self.toolbox_metadata
        }
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        image_info = {}
        if image_path and os.path.isfile(image_path):
            image_info["image_path"] = image_path
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                image_info.update({
                    "width": width,
                    "height": height
                })
            except Exception as e:
                print(f"Error processing image file: {str(e)}")
        return image_info

    def generate_base_response(self, question: str, image: str, max_tokens: int = 2048) -> str:
        image_info = self.get_image_info(image)

        input_data = [question]
        if image_info and "image_path" in image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")


        print("Input data of `generate_base_response()`: ", input_data)
        self.base_response = self.llm_engine(input_data, max_tokens=max_tokens) # default system prompt with question
        # self.base_response = self.llm_engine_fixed(input_data, max_tokens=max_tokens)

        return self.base_response

    def generate_plan(
        self,
        question: str,
        image: Optional[str],
        max_subtasks: int = 5,
        json_data: Any = None,
    ) -> PlanOutline:

        image_info = self.get_image_info(image)

        shared_instructions = f"""
        Task: You are the planning coordinator for DynamicFlow. Decompose the user task into a roadmap for planner-worker collaboration.

        Inputs:
        - Query: {question}
        - Available Tools: {self.available_tools}
        - Tool Metadata: {self.toolbox_metadata}
        - Maximum Number of Subtasks: {max_subtasks}

        Planning Expectations:
        1. Produce an ordered sequence of **1 to {max_subtasks} subtasks**, arranged by logical dependency and difficulty.
        2. Each subtask must have **one actionable, well-defined objective** that can be completed using the tools listed in Available Tools. Reference tool names exactly as provided.
        3. For every subtask, define:
        - **success_criteria**: what conditions mark this step as “done”
        - **handoff_notes**: what the next worker needs to know for smooth continuation
        4. Subtasks should remain focused, practical, and minimally overlapping. Avoid unnecessary complexity.

        Example:
        - Query: "Please identify which column in the CSV file I uploaded is most related to housing prices, and give me a brief analytical summary."
        - Response: 
        {{
        "reasoning": "To determine which column is most correlated with housing price, we must first load the dataset, inspect the schema, compute correlations, and then generate a natural-language summary. Python is suitable for data processing, while the generator tool is needed to produce a readable explanation.",
        "steps": [
            {{
            "step_id": 1,
            "title": "Load and inspect dataset",
            "objective": "Use Python to read the CSV file, list all columns, display data types, and show sample rows.",
            "success_criteria": "Dataset successfully loaded with schema summary and preview available.",
            "suggested_tools": ["Python_Code_Generator_Tool"],
            "handoff_notes": "Provide the list of numerical columns for correlation calculation."
            }},
            {{
            "step_id": 2,
            "title": "Compute correlations with target column",
            "objective": "Use Python to compute Pearson correlation between each numerical column and the 'price' column.",
            "success_criteria": "Sorted list of correlations showing which column has the strongest relationship with price.",
            "suggested_tools": ["Python_Code_Generator_Tool"],
            "handoff_notes": "Provide the highest-correlation column and values to the generator for explanation."
            }},
            {{
            "step_id": 3,
            "title": "Generate interpretation summary",
            "objective": "Use the generator tool to convert the correlation results into a concise explanation describing the main influencing feature of price.",
            "success_criteria": "A clear, coherent summary highlighting the most relevant column and why it correlates with price.",
            "suggested_tools": ["Generalist_Solution_Generator_Tool"],
            "handoff_notes": ""
            }}
        ]
        }}

        Return only valid JSON that conforms to this PlanOutline schema.
        """

        if self.is_multimodal and image_info:
            multimodal_prompt = (
                shared_instructions
                + "\nImage Context: "
                + json.dumps(image_info, ensure_ascii=False)
                + "\nIncorporate visual insights when relevant."
            )
        else:
            multimodal_prompt = shared_instructions

        raw_plan: Any = None

        try:
            if self.is_multimodal and image_info and image_info.get("image_path"):
                with open(image_info["image_path"], "rb") as file:
                    image_bytes = file.read()
                raw_plan = self.llm_engine(
                    [multimodal_prompt, image_bytes],
                    response_format=PlanOutline,
                )
            else:
                raw_plan = self.llm_engine(multimodal_prompt, response_format=PlanOutline)
        except Exception as exc:
            if self.verbose:
                print(f"Planner generate_plan structured call failed: {exc}")

        plan = self._coerce_plan_outline(raw_plan, max_subtasks)

        if json_data is not None:
            json_data["plan_prompt"] = multimodal_prompt
            if isinstance(raw_plan, (str, dict)):
                json_data["plan_raw_response"] = raw_plan
            json_data["plan_response"] = plan.model_dump()

        return plan

    def design_worker(
        self,
        question: str,
        plan_step: PlanStep,
        global_memory: Memory,
        existing_workers: List[Dict[str, Any]],
        image: Optional[str] = None,
        json_data: Any = None,
    ) -> WorkerSpecification:
        memory_snapshot = json.dumps(
            global_memory.get_actions(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        existing_workers_str = json.dumps(
            existing_workers,
            ensure_ascii=False,
            indent=2,
        )
        # available_tools_str = json.dumps(self.available_tools, ensure_ascii=False)
        image_info = self.get_image_info(image)

        prompt = f"""
        Task: You are coordinating workers for DynamicFlow.

        Input:
        - Query: {question}
        - Plan Step {plan_step.step_id}: {plan_step.title}
        - Objective: {plan_step.objective}
        - Success Criteria: {plan_step.success_criteria}
        - Suggested Tools: {plan_step.suggested_tools}
        - Handoff Notes: {plan_step.handoff_notes}
        - Existing Workers: {existing_workers_str}
        - Accumulated Memory: {memory_snapshot}
        - Available Tools: {self.available_tools}

        Design a single worker specialized for this plan step.  
        
        Worker Expectations:
        1. Be actionable and role-specific.
        2. Use tool names strictly as listed in "Available Tools" and choose only the most relevant ones.
        3. Align closely with the success criteria and objective.
        4. Include a precise system prompt that will guide the worker's behavior.

        Example:
        - Input:
            - query = "Please identify which column in the CSV file I uploaded is most related to housing prices, and give me a brief analytical summary."
            - step_id = 1
            - title = "Load and inspect dataset"
            - objective = "Use Python to read the CSV file, list all columns, display data types, and show sample rows."
            - success_criteria = "Dataset successfully loaded with schema summary and preview available."
            - suggested_tools = ["Python_Code_Generator_Tool"]
            - handoff_notes = "Provide the list of numerical columns for correlation calculation."

        - Output:
        {{
        "worker_name": "DatasetInspector",
        "worker_role": "Reads and inspects tabular datasets",
        "mission": "Load the dataset from the provided path, show schema, types, and preview rows.",
        "context": "User needs initial dataset understanding for downstream correlation analysis.",
        "tool_names": ["Python_Code_Generator_Tool"],
        "success_criteria": [
            "Dataset loaded without errors",
            "Columns listed with data types",
            "Sample rows shown",
            "Numerical columns identified for next step"
        ],
        "system_prompt": "You are a dataset inspection worker. Load the dataset, summarize its schema, preview rows, and extract numerical columns."
        }}
        """
        breakpoint()
        def _coerce_spec_construct(raw_spec: Any) -> WorkerSpecification:
            """Convert model output into WorkerSpecification, falling back if parsing fails."""
            parsed_payload = None
            breakpoint()
            txt = raw_spec.strip()
            txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
            
            try:
                parsed_payload = json.loads(txt)
            except Exception as exc:
                if self.verbose:
                    print(f"Planner design_worker: JSON parsing failed: {exc}")

            if parsed_payload is not None:
                try:
                    spec = WorkerSpecification(**parsed_payload)
                    return spec
                except Exception as exc:
                    if self.verbose:
                        print(f"Planner design_worker: WorkerSpecification construction failed: {exc}. Using fallback worker outline.")
            
            spec = WorkerSpecification(
                worker_name=f"worker_step_{plan_step.step_id}",
                worker_role="Specialist",
                mission=plan_step.objective,
                context=f"Carry out plan step {plan_step.step_id} for the query.",
                tool_names=plan_step.suggested_tools,
                success_criteria=[plan_step.success_criteria],
                system_prompt=(
                    "You are a focused DynamicFlow worker. Complete the assigned subtask "
                    "using the provided tools and document outcomes for the planner."
                ),
            )
            return spec

        raw_spec: Any = None

        raw_spec = self.llm_engine(prompt, response_format=WorkerSpecification)
        spec = _coerce_spec_construct(raw_spec)

        if json_data is not None:
            json_data[f"worker_{plan_step.step_id}_design_prompt"] = prompt
            json_data[f"worker_{plan_step.step_id}_design_response"] = spec.model_dump()

        return spec

    def generate_worker_step(
        self,
        question: str,
        image: str,
        plan_step: PlanStep,
        worker_spec: WorkerSpecification,
        worker_memory: Memory,
        global_memory: Memory,
        step_count: int,
        max_step_count: int,
        json_data: Any = None,
    ) -> Any:
        metadata_subset = self._filter_tool_metadata(worker_spec.tool_names)
        worker_memory_snapshot = json.dumps(
            worker_memory.get_actions(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        global_memory_snapshot = json.dumps(
            global_memory.get_actions(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        image_info = self.get_image_info(image)

        prompt = f"""
You are worker {worker_spec.worker_name} ({worker_spec.worker_role}).
Mission: {worker_spec.mission}
Plan Step {plan_step.step_id}: {plan_step.title}
Success Criteria: {plan_step.success_criteria}
Worker-Specific Success Criteria: {worker_spec.success_criteria}
Handoff Notes: {plan_step.handoff_notes}
Image Context: {image_info if image_info else 'None'}

Global Memory Snapshot: {global_memory_snapshot}
Worker Memory Snapshot: {worker_memory_snapshot}

Available Tools: {worker_spec.tool_names}
Tool Metadata: {metadata_subset}
Current Step: {step_count} of {max_step_count}

Decide the optimal next action. Use ONLY the available tools. Respond using the NextStep schema.
"""

        try:
            next_step = self.llm_engine(prompt, response_format=NextStep)
            if json_data is not None:
                key = f"worker_{worker_spec.worker_name}_step_{step_count}_predictor"
                json_data[f"{key}_prompt"] = prompt
                json_data[f"{key}_response"] = next_step.model_dump()
            return next_step
        except Exception as exc:
            if self.verbose:
                print(f"Planner generate_worker_step failed, using fallback: {exc}")
            fallback_tool = worker_spec.tool_names[0] if worker_spec.tool_names else None
            return NextStep(
                justification="Fallback action due to model error.",
                context="Refer to previous outputs and mission description to proceed manually.",
                sub_goal=plan_step.objective,
                tool_name=fallback_tool or "",
            )

    def evaluate_worker_progress(
        self,
        question: str,
        plan_step: PlanStep,
        worker_spec: WorkerSpecification,
        worker_memory: Memory,
        global_memory: Memory,
        json_data: Any = None,
    ) -> WorkerProgressCheck:
        worker_memory_snapshot = json.dumps(
            worker_memory.get_actions(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        global_memory_snapshot = json.dumps(
            global_memory.get_actions(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        prompt = f"""
Evaluate the progress of worker {worker_spec.worker_name} on Plan Step {plan_step.step_id}.

User Query: {question}
Plan Step Objective: {plan_step.objective}
Plan Step Success Criteria: {plan_step.success_criteria}
Worker Success Criteria: {worker_spec.success_criteria}
Worker Memory: {worker_memory_snapshot}
Global Memory: {global_memory_snapshot}

Return a JSON object with fields:
{{
  "analysis": "Detailed assessment",
  "status": "complete" or "continue",
  "blockers": "Any blockers preventing completion",
  "next_action": "Recommended next action"
}}
"""

        try:
            evaluation = self.llm_engine(prompt, response_format=WorkerProgressCheck)
            if json_data is not None:
                key = f"worker_{worker_spec.worker_name}_progress"
                json_data[f"{key}_prompt"] = prompt
                json_data[f"{key}_response"] = evaluation.model_dump()
            return evaluation
        except Exception as exc:
            if self.verbose:
                print(f"Planner evaluate_worker_progress failed, using fallback: {exc}")
            return WorkerProgressCheck(
                analysis="Unable to assess progress due to parsing error.",
                status="continue",
                blockers="Assessment fallback invoked",
                next_action="Planner should manually review worker outputs",
            )

    def summarize_worker_result(
        self,
        question: str,
        plan_step: PlanStep,
        worker_spec: WorkerSpecification,
        worker_memory: Memory,
        global_memory: Memory,
        json_data: Any = None,
    ) -> WorkerSummary:
        worker_memory_snapshot = json.dumps(
            worker_memory.get_actions(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        global_memory_snapshot = json.dumps(
            global_memory.get_actions(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        prompt = f"""
Create a concise summary for plan step {plan_step.step_id} executed by worker {worker_spec.worker_name}.

User Query: {question}
Plan Step Objective: {plan_step.objective}
Worker Success Criteria: {worker_spec.success_criteria}
Worker Memory: {worker_memory_snapshot}
Global Memory: {global_memory_snapshot}

Return a JSON object with fields:
{{
  "summary": "Narrative summary of what was done and found",
  "follow_up": "Notes for the planner or next worker"
}}
"""

        try:
            summary = self.llm_engine(prompt, response_format=WorkerSummary)
            if json_data is not None:
                key = f"worker_{worker_spec.worker_name}_summary"
                json_data[f"{key}_prompt"] = prompt
                json_data[f"{key}_response"] = summary.model_dump()
            return summary
        except Exception as exc:
            if self.verbose:
                print(f"Planner summarize_worker_result failed, using fallback: {exc}")
            return WorkerSummary(
                summary="Worker summary unavailable; planner must synthesize manually.",
                follow_up="Investigate worker outputs manually.",
            )

    def analyze_query(self, question: str, image: str) -> str:
        image_info = self.get_image_info(image)

        if self.is_multimodal:
            query_prompt = f"""
Task: Analyze the given query with accompanying inputs and determine the skills and tools needed to address it effectively.

Available tools: {self.available_tools}

Metadata for the tools: {self.toolbox_metadata}

Image: {image_info}

Query: {question}

Instructions:
1. Carefully read and understand the query and any accompanying inputs.
2. Identify the main objectives or tasks within the query.
3. List the specific skills that would be necessary to address the query comprehensively.
4. Examine the available tools in the toolbox and determine which ones might relevant and useful for addressing the query. Make sure to consider the user metadata for each tool, including limitations and potential applications (if available).
5. Provide a brief explanation for each skill and tool you've identified, describing how it would contribute to answering the query.

Your response should include:
1. A concise summary of the query's main points and objectives, as well as content in any accompanying inputs.
2. A list of required skills, with a brief explanation for each.
3. A list of relevant tools from the toolbox, with a brief explanation of how each tool would be utilized and its potential limitations.
4. Any additional considerations that might be important for addressing the query effectively.

Please present your analysis in a clear, structured format.
                        """
        else: 
            query_prompt = f"""
Task: Analyze the given query to determine necessary skills and tools.

Inputs:
- Query: {question}
- Available tools: {self.available_tools}
- Metadata for tools: {self.toolbox_metadata}

Instructions:
1. Identify the main objectives in the query.
2. List the necessary skills and tools.
3. For each skill and tool, explain how it helps address the query.
4. Note any additional considerations.

Format your response with a summary of the query, lists of skills and tools with explanations, and a section for additional considerations.

Be biref and precise with insight. 
"""


        input_data = [query_prompt]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        print("Input data of `analyze_query()`: ", input_data)

        # self.query_analysis = self.llm_engine_mm(input_data, response_format=QueryAnalysis)
        # self.query_analysis = self.llm_engine(input_data, response_format=QueryAnalysis)
        self.query_analysis = self.llm_engine_fixed(input_data, response_format=QueryAnalysis) # default dashscope don't need response_format, but openai supports it

        return str(self.query_analysis).strip()

    def extract_context_subgoal_and_tool(
        self,
        response: Any,
        allowed_tools: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        tools_reference = allowed_tools if allowed_tools is not None else self.available_tools
        try:
            if isinstance(response, str):
                # Attempt to parse the response as JSON
                try:
                    response_dict = json.loads(response)
                    response = NextStep(**response_dict)
                except Exception as e:
                    print(f"Failed to parse response as JSON: {str(e)}")
            if isinstance(response, NextStep):
                print("arielg 1")
                context = response.context.strip()
                sub_goal = response.sub_goal.strip()
                tool_name = response.tool_name.strip()
            else:
                print("arielg 2")
                text = response.replace("**", "")

                # Pattern to match the exact format
                pattern = r"Context:\s*(.*?)Sub-Goal:\s*(.*?)Tool Name:\s*(.*?)\s*(?:```)?\s*(?=\n\n|\Z)"

                # Find all matches
                matches = re.findall(pattern, text, re.DOTALL)

                # Return the last match (most recent/relevant)
                context, sub_goal, tool_name = matches[-1]
                context = context.strip()
                sub_goal = sub_goal.strip()

            if tools_reference and tool_name not in tools_reference:
                return context, sub_goal, None
        except Exception as e:
            print(f"Error extracting context, sub-goal, and tool name: {str(e)}")
            return None, None, None

        return context, sub_goal, tool_name

    def generate_next_step(self, question: str, image: str, query_analysis: str, memory: Memory, step_count: int, max_step_count: int, json_data: Any = None) -> Any:
        if self.is_multimodal:
            prompt_generate_next_step = f"""
Task: Determine the optimal next step to address the given query based on the provided analysis, available tools, and previous steps taken.

Context:
Query: {question}
Image: {image}
Query Analysis: {query_analysis}

Available Tools:
{self.available_tools}

Tool Metadata:
{self.toolbox_metadata}

Previous Steps and Their Results:
{memory.get_actions()}

Current Step: {step_count} in {max_step_count} steps
Remaining Steps: {max_step_count - step_count}

Instructions:
1. Analyze the context thoroughly, including the query, its analysis, any image, available tools and their metadata, and previous steps taken.

2. Determine the most appropriate next step by considering:
- Key objectives from the query analysis
- Capabilities of available tools
- Logical progression of problem-solving
- Outcomes from previous steps
- Current step count and remaining steps

3. Select ONE tool best suited for the next step, keeping in mind the limited number of remaining steps.

4. Formulate a specific, achievable sub-goal for the selected tool that maximizes progress towards answering the query.

Response Format:
Your response MUST follow this structure:
1. Justification: Explain your choice in detail.
2. Context, Sub-Goal, and Tool: Present the context, sub-goal, and the selected tool ONCE with the following format:

Context: <context>
Sub-Goal: <sub_goal>
Tool Name: <tool_name>

Where:
- <context> MUST include ALL necessary information for the tool to function, structured as follows:
* Relevant data from previous steps
* File names or paths created or used in previous steps (list EACH ONE individually)
* Variable names and their values from previous steps' results
* Any other context-specific information required by the tool
- <sub_goal> is a specific, achievable objective for the tool, based on its metadata and previous outcomes.
It MUST contain any involved data, file names, and variables from Previous Steps and Their Results that the tool can act upon.
- <tool_name> MUST be the exact name of a tool from the available tools list.

Rules:
- Select only ONE tool for this step.
- The sub-goal MUST directly address the query and be achievable by the selected tool.
- The Context section MUST include ALL necessary information for the tool to function, including ALL relevant file paths, data, and variables from previous steps.
- The tool name MUST exactly match one from the available tools list: {self.available_tools}.
- Avoid redundancy by considering previous steps and building on prior results.
- Your response MUST conclude with the Context, Sub-Goal, and Tool Name sections IN THIS ORDER, presented ONLY ONCE.
- Include NO content after these three sections.

Example (do not copy, use only as reference):
Justification: [Your detailed explanation here]
Context: Image path: "example/image.jpg", Previous detection results: [list of objects]
Sub-Goal: Detect and count the number of specific objects in the image "example/image.jpg"
Tool Name: Object_Detector_Tool

Remember: Your response MUST end with the Context, Sub-Goal, and Tool Name sections, with NO additional content afterwards.
                        """
        else:
            prompt_generate_next_step = f"""
Task: Determine the optimal next step to address the query using available tools and previous steps.

Context:
- **Query:** {question}
- **Query Analysis:** {query_analysis}
- **Available Tools:** {self.available_tools}
- **Toolbox Metadata:** {self.toolbox_metadata}
- **Previous Steps:** {memory.get_actions()}

Instructions:
1. Analyze the query, previous steps, and available tools.
2. Select the **single best tool** for the next step.
3. Formulate a specific, achievable **sub-goal** for that tool.
4. Provide all necessary **context** (data, file names, variables) for the tool to function.

Response Format:
1.  **Justification:** Explain your choice of tool and sub-goal.
2.  **Context:** Provide all necessary information for the tool.
3.  **Sub-Goal:** State the specific objective for the tool.
4.  **Tool Name:** State the exact name of the selected tool.

Rules:
- Select only ONE tool.
- The sub-goal must be directly achievable by the selected tool.
- The Context section must contain all information the tool needs to function.
- The response must end with the Context, Sub-Goal, and Tool Name sections in that order, with no extra content.
                    """
            
        next_step = self.llm_engine(prompt_generate_next_step, response_format=NextStep)
        if json_data is not None:
            json_data[f"action_predictor_{step_count}_prompt"] = prompt_generate_next_step
            json_data[f"action_predictor_{step_count}_response"] = str(next_step)
        return next_step

    def verificate_context(self, question: str, image: str, query_analysis: str, memory: Memory, step_count: int = 0, json_data: Any = None) -> Any:
        image_info = self.get_image_info(image)
        if self.is_multimodal:
            prompt_memory_verification = f"""
Task: Thoroughly evaluate the completeness and accuracy of the memory for fulfilling the given query, considering the potential need for additional tool usage.

Context:
Query: {question}
Image: {image_info}
Available Tools: {self.available_tools}
Toolbox Metadata: {self.toolbox_metadata}
Initial Analysis: {query_analysis}
Memory (tools used and results): {memory.get_actions()}

Detailed Instructions:
1. Carefully analyze the query, initial analysis, and image (if provided):
   - Identify the main objectives of the query.
   - Note any specific requirements or constraints mentioned.
   - If an image is provided, consider its relevance and what information it contributes.

2. Review the available tools and their metadata:
   - Understand the capabilities and limitations and best practices of each tool.
   - Consider how each tool might be applicable to the query.

3. Examine the memory content in detail:
   - Review each tool used and its execution results.
   - Assess how well each tool's output contributes to answering the query.

4. Critical Evaluation (address each point explicitly):
   a) Completeness: Does the memory fully address all aspects of the query?
      - Identify any parts of the query that remain unanswered.
      - Consider if all relevant information has been extracted from the image (if applicable).

   b) Unused Tools: Are there any unused tools that could provide additional relevant information?
      - Specify which unused tools might be helpful and why.

   c) Inconsistencies: Are there any contradictions or conflicts in the information provided?
      - If yes, explain the inconsistencies and suggest how they might be resolved.

   d) Verification Needs: Is there any information that requires further verification due to tool limitations?
      - Identify specific pieces of information that need verification and explain why.

   e) Ambiguities: Are there any unclear or ambiguous results that could be clarified by using another tool?
      - Point out specific ambiguities and suggest which tools could help clarify them.

5. Final Determination:
   Based on your thorough analysis, decide if the memory is complete and accurate enough to generate the final output, or if additional tool usage is necessary.

Response Format:

If the memory is complete, accurate, AND verified:
Explanation:
<Provide a detailed explanation of why the memory is sufficient. Reference specific information from the memory and explain its relevance to each aspect of the task. Address how each main point of the query has been satisfied.>

Conclusion: STOP

If the memory is incomplete, insufficient, or requires further verification:
Explanation:
<Explain in detail why the memory is incomplete. Identify specific information gaps or unaddressed aspects of the query. Suggest which additional tools could be used, how they might contribute, and why their input is necessary for a comprehensive response.>

Conclusion: CONTINUE

IMPORTANT: Your response MUST end with either 'Conclusion: STOP' or 'Conclusion: CONTINUE' and nothing else. Ensure your explanation thoroughly justifies this conclusion.
"""
        else:
            prompt_memory_verification = f"""
Task: Evaluate if the current memory is complete and accurate enough to answer the query, or if more tools are needed.

Context:
- **Query:** {question}
- **Available Tools:** {self.available_tools}
- **Toolbox Metadata:** {self.toolbox_metadata}
- **Initial Analysis:** {query_analysis}
- **Memory (Tools Used & Results):** {memory.get_actions()}

Instructions:
1.  Review the query, initial analysis, and memory.
2.  Assess the completeness of the memory: Does it fully address all parts of the query?
3.  Check for potential issues:
    -   Are there any inconsistencies or contradictions?
    -   Is any information ambiguous or in need of verification?
4.  Determine if any unused tools could provide missing information.

Final Determination:
-   If the memory is sufficient, explain why and conclude with "STOP".
-   If more information is needed, explain what's missing, which tools could help, and conclude with "CONTINUE".

IMPORTANT: The response must end with either "Conclusion: STOP" or "Conclusion: CONTINUE".
"""

        input_data = [prompt_memory_verification]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        # stop_verification = self.llm_engine_mm(input_data, response_format=MemoryVerification)
        stop_verification = self.llm_engine_fixed(input_data, response_format=MemoryVerification)
        # stop_verification = self.llm_engine(input_data, response_format=MemoryVerification)
        if json_data is not None:
            json_data[f"verifier_{step_count}_prompt"] = input_data
            json_data[f"verifier_{step_count}_response"] = str(stop_verification)
        return stop_verification

    def extract_conclusion(self, response: Any) -> tuple:
        if isinstance(response, str):
            # Attempt to parse the response as JSON
            try:
                response_dict = json.loads(response)
                response = MemoryVerification(**response_dict)
            except Exception as e:
                print(f"Failed to parse response as JSON: {str(e)}")
        if isinstance(response, MemoryVerification):
            analysis = response.analysis
            stop_signal = response.stop_signal
            if stop_signal:
                return analysis, 'STOP'
            else:
                return analysis, 'CONTINUE'
        else:
            analysis = response
            pattern = r'conclusion\**:?\s*\**\s*(\w+)'
            matches = list(re.finditer(pattern, response, re.IGNORECASE | re.DOTALL))
            # if match:
            #     conclusion = match.group(1).upper()
            #     if conclusion in ['STOP', 'CONTINUE']:
            #         return conclusion
            if matches:
                conclusion = matches[-1].group(1).upper()
                if conclusion in ['STOP', 'CONTINUE']:
                    return analysis, conclusion

            # If no valid conclusion found, search for STOP or CONTINUE anywhere in the text
            if 'stop' in response.lower():
                return analysis, 'STOP'
            elif 'continue' in response.lower():
                return analysis, 'CONTINUE'
            else:
                print("No valid conclusion (STOP or CONTINUE) found in the response. Continuing...")
                return analysis, 'CONTINUE'

    def generate_final_output(self, question: str, image: str, memory: Memory) -> str:
        image_info = self.get_image_info(image)
        if self.is_multimodal:
            prompt_generate_final_output = f"""
Task: Generate the final output based on the query, image, and tools used in the process.

Context:
Query: {question}
Image: {image_info}
Actions Taken:
{memory.get_actions()}

Instructions:
1. Review the query, image, and all actions taken during the process.
2. Consider the results obtained from each tool execution.
3. Incorporate the relevant information from the memory to generate the step-by-step final output.
4. The final output should be consistent and coherent using the results from the tools.

Output Structure:
Your response should be well-organized and include the following sections:

1. Summary:
   - Provide a brief overview of the query and the main findings.

2. Detailed Analysis:
   - Break down the process of answering the query step-by-step.
   - For each step, mention the tool used, its purpose, and the key results obtained.
   - Explain how each step contributed to addressing the query.

3. Key Findings:
   - List the most important discoveries or insights gained from the analysis.
   - Highlight any unexpected or particularly interesting results.

4. Answer to the Query:
   - Directly address the original question with a clear and concise answer.
   - If the query has multiple parts, ensure each part is answered separately.

5. Additional Insights (if applicable):
   - Provide any relevant information or insights that go beyond the direct answer to the query.
   - Discuss any limitations or areas of uncertainty in the analysis.

6. Conclusion:
   - Summarize the main points and reinforce the answer to the query.
   - If appropriate, suggest potential next steps or areas for further investigation.
"""
        else:
                prompt_generate_final_output = f"""
Task: Generate the final output based on the query and the results from all tools used.

Context:
- **Query:** {question}
- **Actions Taken:** {memory.get_actions()}

Instructions:
1. Review the query and the results from all tool executions.
2. Incorporate the relevant information to create a coherent, step-by-step final output.
"""

        input_data = [prompt_generate_final_output]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        # final_output = self.llm_engine_mm(input_data)
        # final_output = self.llm_engine(input_data)
        final_output = self.llm_engine_fixed(input_data)

        return final_output


    def generate_direct_output(self, question: str, image: str, memory: Memory) -> str:
        image_info = self.get_image_info(image)
        if self.is_multimodal:
            prompt_generate_final_output = f"""
Context:
Query: {question}
Image: {image_info}
Initial Analysis:
{self.query_analysis}
Actions Taken:
{memory.get_actions()}

Please generate the concise output based on the query, image information, initial analysis, and actions taken. Break down the process into clear, logical, and conherent steps. Conclude with a precise and direct answer to the query.

Answer:
"""
        else:
            prompt_generate_final_output = f"""
Task: Generate a concise final answer to the query based on all provided context.

Context:
- **Query:** {question}
- **Initial Analysis:** {self.query_analysis}
- **Actions Taken:** {memory.get_actions()}

Instructions:
1. Review the query and the results from all actions.
2. Synthesize the key findings into a clear, step-by-step summary of the process.
3. Provide a direct, precise answer to the original query.

Output Structure:
1.  **Process Summary:** A clear, step-by-step breakdown of how the query was addressed, including the purpose and key results of each action.
2.  **Answer:** A direct and concise final answer to the query.
"""

        input_data = [prompt_generate_final_output]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        # final_output = self.llm_engine(input_data)
        final_output = self.llm_engine_fixed(input_data)
        # final_output = self.llm_engine_mm(input_data)

        return final_output

