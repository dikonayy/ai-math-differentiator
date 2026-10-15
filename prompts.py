"""
All prompts sent to the LLM live here, so the differentiation strategy can be
read (and graded) in one place.

Structure:
  SYSTEM_PROMPT          - shared rules: tutor role, ground-truth rule, formatting
  LEVELS[level]          - the differentiation profile for each learner level
  build_task_prompt()    - asks the model to generate a task for one level
  build_feedback_prompt()- asks for feedback on a student's answer
  build_chat_prompt()    - answers a free-form question about the current task
"""
from __future__ import annotations

from typing import Dict, List, Optional

from math_engine import Quadratic, format_root

SYSTEM_PROMPT = """You are a patient, encouraging mathematics tutor inside a web app for \
differentiated learning. The topic is solving quadratic equations.

Hard rules:
1. GROUND TRUTH. The discriminant and the roots were computed by a computer algebra system \
and are given to you as GROUND TRUTH. Never compute or state different values. If your own \
arithmetic disagrees, the ground truth is right.
2. Adapt everything (vocabulary, amount of scaffolding, hint style, depth of explanation) \
to the LEARNER PROFILE you are given. The profile matters more than your default style.
3. Math formatting: write every formula in LaTeX between single dollar signs, e.g. \
$x^2 - 5x + 6 = 0$. Use $$...$$ only for a formula on its own line. Never use \\( \\) or \\[ \\].
4. Write in the language requested by the user prompt. Keep LaTeX itself in standard notation.
5. Be concise. Students read this on a phone."""


LEVELS: Dict[str, Dict[str, str]] = {
    "foundation": {
        "name": "Level 1: Foundation",
        "learner": "A learner who is new to quadratic equations or finds them hard. "
        "Needs short sentences, everyday words, every term defined, and no skipped steps.",
        "task": "Use the equation shown to the student exactly as given (the app displays it above "
        "your statement, so refer to it as 'the equation above' instead of retyping it). Turn the task "
        "into a guided worksheet: 4-5 numbered mini-steps the student fills in one by one: "
        "(1) write down a, b and c, (2) compute the discriminant $D = b^2 - 4ac$, (3) decide how many "
        "roots there are, (4) substitute into $x = \\frac{-b \\pm \\sqrt{D}}{2a}$, (5) check by substitution. "
        "Remind them of the formulas they need inside the steps. No real-world story.",
        "hints": "Very direct. Hint 1 names the exact next action. Hint 2 shows the formula with this "
        "equation's numbers substituted. Hint 3 shows the arithmetic up to the last step but does not "
        "state the final roots.",
        "solution": "Write out every single arithmetic operation. Explicitly warn about the most common "
        "mistake for this equation (usually the sign of b or forgetting the ±).",
        "feedback": "Point directly at the most likely mistake and tell the student which mini-step to "
        "redo and how. Be warm and reassuring.",
    },
    "standard": {
        "name": "Level 2: Standard",
        "learner": "A typical student at grade level who knows the quadratic formula and factoring "
        "and should practise solving independently.",
        "task": "The equation is deliberately shown in a non-standard form (the app displays it above "
        "your statement; do not retype it). Write a short textbook-style instruction such as 'Solve the "
        "equation and give all real roots.' Do NOT break it into steps and do NOT mention that it must "
        "be rearranged; recognising that is part of the task.",
        "hints": "Conceptual nudges, no numbers. Hint 1: what form must the equation be in first. "
        "Hint 2: which method suits it (factoring, Vieta or the formula) and why. Hint 3: what the "
        "discriminant tells you here.",
        "solution": "A concise textbook solution: rearrange to standard form, simplify by a common "
        "factor if possible, solve, check. If the equation factors nicely, show factoring as an "
        "alternative method in one line.",
        "feedback": "Name the stage where the work went wrong (rearranging, dividing out a factor, the "
        "discriminant, the formula) without giving any numbers from the solution.",
    },
    "advanced": {
        "name": "Level 3: Advanced",
        "learner": "A strong student who is ready for modelling: translating a real situation into an "
        "equation, solving it and interpreting the result critically.",
        "task": "Write a realistic multi-step word problem (geometry/area, projectile motion, business "
        "revenue, travel speed and time, or similar; choose something fresh) whose mathematical model is "
        "EXACTLY the given equation or a non-zero multiple of it. Do NOT show the equation in the "
        "statement; the student must build it. Define what the unknown represents and include a unit. "
        "If one root has no meaning in the story (e.g. negative length), the student must notice and "
        "reject it. Ask one short extension question at the end (e.g. 'What if...?') that does not "
        "change the main answer.",
        "hints": "Socratic questions only, no formulas given away. Hint 1: what should the unknown be? "
        "Hint 2: what relationship in the story gives an equation? Hint 3: which solutions make sense "
        "in this context and why?",
        "solution": "Model → equation → simplify to standard form → solve → interpret each root in "
        "context → check against the story → short answer to the extension question.",
        "feedback": "Ask one guiding question that makes the student find the error themselves "
        "(modelling, algebra or interpretation). Encourage checking the answer against the story.",
    },
}

LEVEL_KEYS = list(LEVELS.keys())


def _ground_truth(q: Quadratic) -> str:
    roots = q.roots
    if not roots:
        roots_text = "no real roots (D < 0)"
    else:
        roots_text = ", ".join(f"${format_root(r)}$" for r in roots)
    return (
        "GROUND TRUTH (from SymPy, do not contradict):\n"
        f"- canonical equation: ${q.latex}$  (a = {q.a}, b = {q.b}, c = {q.c})\n"
        f"- discriminant: D = {q.discriminant}\n"
        f"- real roots: {roots_text}"
    )


def _profile(level: str) -> str:
    p = LEVELS[level]
    return (
        f"LEARNER PROFILE: {p['name']}\n"
        f"- Who the learner is: {p['learner']}\n"
        f"- How to write the task: {p['task']}\n"
        f"- How to write the 3 hints: {p['hints']}\n"
        f"- How to write the solution: {p['solution']}"
    )


def build_task_prompt(
    q: Quadratic,
    level: str,
    shown_latex: Optional[str],
    language: str,
    previous_errors: Optional[List[str]] = None,
) -> str:
    shown = (
        f"The student sees this equation: ${shown_latex}$"
        if shown_latex
        else "The student does NOT see any equation; they must build it from your story."
    )
    if level == "advanced":
        roots_rule = (
            "In `final_roots` list ONLY the roots that are meaningful in your story (at least one). "
            "In `model_a`, `model_b`, `model_c` give the coefficients of the equation your story leads "
            "to when written as model_a*x^2 + model_b*x + model_c = 0. It must be the canonical "
            "equation multiplied by a non-zero number, so double-check your story's numbers."
        )
    else:
        roots_rule = (
            "In `final_roots` list ALL real roots (empty list if there are none). "
            "Set model_a, model_b, model_c to the canonical a, b, c."
        )
    errors = ""
    if previous_errors:
        errors = (
            "\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED by the automatic checker:\n- "
            + "\n- ".join(previous_errors)
            + "\nFix these problems."
        )
    return f"""{_ground_truth(q)}

{_profile(level)}

{shown}

Generate the task in {language}. Return JSON with:
- `title`: a short title for the task
- `statement`: the task text the student reads (Markdown + LaTeX)
- `hints`: exactly 3 progressive hints, from gentlest to strongest, none revealing the final answer
- `solution_steps`: the full worked solution as a list of steps
- `final_answer`: one sentence stating the answer (with unit on the Advanced level)
- `final_roots`: each root as a plain expression like `3`, `-1/2` or `(5+sqrt(13))/2`, no LaTeX
- `model_a`, `model_b`, `model_c`: numbers

{roots_rule}{errors}"""


# The checker deliberately gets NO ground truth: it must solve the story on its own,
# so a story whose numbers don't lead to the intended equation is caught.
SOLVER_SYSTEM_PROMPT = """You are a careful mathematics examiner. You receive a word problem \
and solve it independently. Translate the story into an equation literally, using exactly the \
numbers and relationships stated. Do not guess what the author intended."""


def build_solver_prompt(statement: str) -> str:
    return f"""Word problem:
\"\"\"{statement}\"\"\"

Model the MAIN question (ignore any extension question). Return JSON with:
- `unknown`: what x stands for, with unit
- `equation_lhs`, `equation_rhs`: the two sides of the equation in x that the story gives, \
as plain expressions using only numbers, x, + - * / ^ and parentheses (e.g. `x*(10 - 2*x)` and `16`)
- `answer_roots`: the value(s) of x that answer the question, after rejecting values that make \
no sense in the story, as plain expressions like `3` or `(5+sqrt(13))/2`"""


def build_feedback_prompt(
    q: Quadratic,
    level: str,
    task_statement: str,
    student_answer: str,
    check_status: str,
    language: str,
) -> str:
    verdicts = {
        "correct": "CORRECT. Congratulate briefly (one sentence), then give one short insight or "
        "follow-up challenge suited to the learner level.",
        "partial": "PARTIALLY CORRECT: some roots are right, but a root is missing or an extra one is "
        "present. Do not say which value is right; help the student find what is missing or extra.",
        "wrong": "WRONG. Do NOT reveal the correct roots. Diagnose the most likely mistake from the "
        "student's answer and respond as the feedback style says.",
        "unparsed": "The checker could not read the answer. Explain briefly how to type it: numbers "
        "separated by ';', fractions like -1/2, roots like sqrt(13) or (5+sqrt(13))/2, or 'no roots'.",
    }
    return f"""{_ground_truth(q)}

LEARNER PROFILE: {LEVELS[level]['name']}
- Who the learner is: {LEVELS[level]['learner']}
- Feedback style: {LEVELS[level]['feedback']}

The task the student is solving:
\"\"\"{task_statement}\"\"\"

The student's answer: \"\"\"{student_answer}\"\"\"
Automatic checker verdict: {verdicts[check_status]}

Write the feedback in {language}, at most 4 sentences, addressed directly to the student."""


def build_chat_prompt(
    q: Quadratic,
    level: str,
    task_statement: str,
    history: List[Dict[str, str]],
    question: str,
    solution_revealed: bool,
    language: str,
) -> str:
    reveal_rule = (
        "The student has already opened the full solution, so you may discuss the answer freely."
        if solution_revealed
        else "Do NOT state the final roots; guide the student towards them instead."
    )
    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-6:])
    return f"""{_ground_truth(q)}

LEARNER PROFILE: {LEVELS[level]['name']}
- Who the learner is: {LEVELS[level]['learner']}
- Explanation style: {LEVELS[level]['feedback']}

The task the student is working on:
\"\"\"{task_statement}\"\"\"

{reveal_rule}

Conversation so far:
{transcript or '(none)'}

STUDENT: {question}

Answer in {language}, matched to the learner level, in at most 6 sentences. If the question is \
unrelated to this math task, politely steer back to the task."""
