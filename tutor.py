"""
Gemini wrapper: generates level-specific tasks and feedback, and refuses to
show a task until SymPy has confirmed its math.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

import prompts
from math_engine import Quadratic, equation_matches, is_same_equation, parse_roots, roots_match

MAX_ATTEMPTS = 3


class TaskPayload(BaseModel):
    """JSON schema the model must follow (Gemini structured output)."""

    title: str
    statement: str
    hints: List[str] = Field(description="Exactly 3 progressive hints")
    solution_steps: List[str]
    final_answer: str
    final_roots: List[str]
    model_a: float
    model_b: float
    model_c: float


class SolverPayload(BaseModel):
    """What an independent solver derives from an Advanced word problem."""

    unknown: str
    equation_lhs: str
    equation_rhs: str
    answer_roots: List[str]


@dataclass
class GeneratedTask:
    level: str
    equation: Quadratic
    shown_latex: Optional[str]
    payload: TaskPayload
    attempts: int


class TaskGenerationError(RuntimeError):
    pass


def verify_task(payload: TaskPayload, q: Quadratic, level: str) -> List[str]:
    """Return a list of problems; an empty list means the task passed every check."""
    problems = []
    if len(payload.hints) != 3:
        problems.append(f"`hints` must contain exactly 3 items, got {len(payload.hints)}.")
    if not is_same_equation(q, payload.model_a, payload.model_b, payload.model_c):
        problems.append(
            f"Your model equation {payload.model_a}x^2 + {payload.model_b}x + {payload.model_c} = 0 "
            f"is not a multiple of the canonical equation {q.plain}."
        )
    if not roots_match(payload.final_roots, q, allow_subset=(level == "advanced")):
        problems.append(
            f"`final_roots` {payload.final_roots} do not match the ground-truth roots "
            f"{[str(r) for r in q.roots]}."
        )
    return problems


def verify_story(solver: SolverPayload, q: Quadratic, final_roots: List[str]) -> List[str]:
    """Check that the story itself (not just what the AI claims about it) leads to the equation."""
    if not equation_matches(q, solver.equation_lhs, solver.equation_rhs):
        return [
            f"An independent solver read your story and got the equation "
            f"{solver.equation_lhs} = {solver.equation_rhs}, which is NOT equivalent to {q.plain}. "
            "Change the numbers or relationships in the story so it really leads to the canonical equation."
        ]
    solver_roots = parse_roots("; ".join(solver.answer_roots)) or []
    claimed = parse_roots("; ".join(final_roots)) or []
    if sorted(round(v, 2) for v in solver_roots) != sorted(round(v, 2) for v in claimed):
        return [
            f"An independent solver answered {solver.answer_roots} but you said {final_roots}. "
            "Make the question state clearly which solutions are acceptable."
        ]
    return []


class Tutor:
    def __init__(self, api_key: str, model: str):
        # Gemini often answers 503 "high demand" for a moment: retry with backoff before giving up
        retry = types.HttpRetryOptions(attempts=4, initial_delay=2, max_delay=20, http_status_codes=[429, 500, 503, 504])
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(retry_options=retry))
        self.model = model

    def _generate(self, prompt: str, temperature: float, schema=None, system: Optional[str] = None):
        config = types.GenerateContentConfig(
            system_instruction=system or prompts.SYSTEM_PROMPT,
            temperature=temperature,
        )
        if schema is not None:
            config.response_mime_type = "application/json"
            config.response_schema = schema
        return self.client.models.generate_content(model=self.model, contents=prompt, config=config)

    def generate_task(
        self, q: Quadratic, level: str, shown_latex: Optional[str], language: str
    ) -> GeneratedTask:
        errors: List[str] = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            prompt = prompts.build_task_prompt(q, level, shown_latex, language, errors or None)
            payload = self._parse(self._generate(prompt, temperature=0.9, schema=TaskPayload), TaskPayload)
            errors = verify_task(payload, q, level)
            if not errors and level == "advanced":
                errors = verify_story(self.solve_independently(payload.statement), q, payload.final_roots)
            if not errors:
                return GeneratedTask(level, q, shown_latex, payload, attempt)
        raise TaskGenerationError(
            "The AI could not produce a mathematically verified task in "
            f"{MAX_ATTEMPTS} attempts. Last problems: " + " ".join(errors)
        )

    def solve_independently(self, statement: str) -> SolverPayload:
        """Second opinion without ground truth: catches stories whose numbers don't fit the equation."""
        response = self._generate(
            prompts.build_solver_prompt(statement),
            temperature=0.0,
            schema=SolverPayload,
            system=prompts.SOLVER_SYSTEM_PROMPT,
        )
        return self._parse(response, SolverPayload)

    @staticmethod
    def _parse(response, schema):
        return response.parsed if response.parsed is not None else schema.model_validate_json(response.text)

    def feedback(
        self, task: GeneratedTask, student_answer: str, check_status: str, language: str
    ) -> str:
        prompt = prompts.build_feedback_prompt(
            task.equation, task.level, task.payload.statement, student_answer, check_status, language
        )
        return self._generate(prompt, temperature=0.4).text

    def chat(
        self,
        task: GeneratedTask,
        history: List[Dict[str, str]],
        question: str,
        solution_revealed: bool,
        language: str,
    ) -> str:
        prompt = prompts.build_chat_prompt(
            task.equation,
            task.level,
            task.payload.statement,
            history,
            question,
            solution_revealed,
            language,
        )
        return self._generate(prompt, temperature=0.5).text
