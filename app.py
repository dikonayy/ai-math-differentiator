"""
AI Math Differentiator - Streamlit front end.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import os
import random
from typing import Optional

import streamlit as st

import prompts
from math_engine import Quadratic, check_answer, format_root, presented_latex, random_quadratic, select_roots
from tutor import GeneratedTask, TaskGenerationError, Tutor

st.set_page_config(page_title="AI Math Differentiator", page_icon="🧮", layout="centered")

LANGUAGES = {"English": "English", "Русский": "Russian", "Қазақша": "Kazakh"}
LEVEL_CAPTIONS = {
    "foundation": "Guided step-by-step, direct hints",
    "standard": "Grade-level, solve independently",
    "advanced": "Real-world modelling, multi-step",
}
# Fallback chain: each model has its own free-tier quota
DEFAULT_MODELS = "gemini-flash-latest,gemini-3.5-flash,gemini-flash-lite-latest,gemini-3.1-flash-lite"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def get_setting(name: str, default: str = "") -> str:
    """Read from environment first, then from .streamlit/secrets.toml (Streamlit Cloud secrets)."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return str(st.secrets[name])
    except Exception:
        return default


@st.cache_resource
def get_tutor(api_key: str, models: str) -> Tutor:
    return Tutor(api_key, models)


def friendly_error(e: Exception) -> str:
    code = getattr(e, "code", None)
    if code == 429:
        return "The AI quota is used up on all models for now (Gemini free tier limit). Please wait a minute and try again."
    if code in (401, 403):
        return "The Gemini API key was rejected. Check GEMINI_API_KEY in the app secrets."
    if code in (500, 503, 504):
        return "Gemini is overloaded right now (this happens on the free tier). Please try again in a minute."
    if code == 404:
        return "No configured Gemini model is available. Set GEMINI_MODEL in the app secrets to available models."
    return f"The AI service returned an error: {e}"


state = st.session_state
state.setdefault("equation", random_quadratic())
state.setdefault("tasks", {})  # (a, b, c, level, language) -> GeneratedTask
state.setdefault("progress", {})  # same key -> hints shown, attempts, chat, ...

api_key = get_setting("GEMINI_API_KEY")
models = get_setting("GEMINI_MODEL", DEFAULT_MODELS)


# ---------------------------------------------------------------------------
# Sidebar: the core problem
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("1. Core problem")
    st.caption("One quadratic equation $ax^2 + bx + c = 0$ is the base for all three levels.")

    source = st.radio("Equation source", ["Random", "Enter my own"], horizontal=True)
    if source == "Random":
        if st.button("🎲 New random equation", width="stretch"):
            state.equation = random_quadratic()
    else:
        cols = st.columns(3)
        a = cols[0].number_input("a", value=1, step=1, format="%d")
        b = cols[1].number_input("b", value=-5, step=1, format="%d")
        c = cols[2].number_input("c", value=6, step=1, format="%d")
        if st.button("Use this equation", width="stretch"):
            if a == 0:
                st.error("a cannot be 0, otherwise the equation is not quadratic.")
            else:
                state.equation = Quadratic(int(a), int(b), int(c))

    q: Quadratic = state.equation
    st.latex(q.latex)

    st.header("2. Settings")
    language_label = st.selectbox("AI language", list(LANGUAGES))
    language = LANGUAGES[language_label]

    with st.expander("How it works"):
        st.markdown(
            "- **Gemini** rewrites the same equation for each learner level and gives hints, "
            "feedback and explanations in the style of that level.\n"
            "- **SymPy** (a computer algebra system) computes the true roots. Every AI-generated "
            "task is checked against them and regenerated if it is wrong, and your answers are "
            "graded by SymPy, not by the AI."
        )


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🧮 AI Math Differentiator")
st.caption("One quadratic equation, three learning levels. AI adapts the task, SymPy guarantees the math.")

if not api_key:
    st.error(
        "GEMINI_API_KEY is not set. Add it to `.streamlit/secrets.toml` locally, "
        "or to **App settings → Secrets** on Streamlit Cloud. See README.md."
    )
    st.stop()

tutor = get_tutor(api_key, models)

level = st.radio(
    "Choose your level",
    prompts.LEVEL_KEYS,
    format_func=lambda k: prompts.LEVELS[k]["name"],
    captions=[LEVEL_CAPTIONS[k] for k in prompts.LEVEL_KEYS],
    horizontal=True,
)

key = (q.a, q.b, q.c, level, language)
task: Optional[GeneratedTask] = state.tasks.get(key)

if level == "advanced" and not q.roots:
    st.warning(
        "This equation has no real roots, so it cannot describe a real-world situation. "
        "Pick another equation for the Advanced level."
    )
    st.stop()


def generate() -> bool:
    """Ask the AI for a verified task; returns False (and shows why) if that failed."""
    try:
        with st.spinner("The AI is adapting the task and SymPy is checking it…"):
            shown = presented_latex(q, level, random.Random())
            state.tasks[key] = tutor.generate_task(q, level, shown, language)
            state.progress[key] = {
                "hints": 0, "attempts": 0, "solution": False, "result": None, "chat": [],
            }
        return True
    except TaskGenerationError as e:
        st.error(str(e))
    except Exception as e:  # network / quota / auth problems
        st.error(friendly_error(e))
    return False


if task is None:
    st.info(f"No task yet for **{prompts.LEVELS[level]['name']}**.")
    if st.button("✨ Generate task", type="primary") and generate():
        st.rerun()
    st.stop()

progress = state.progress[key]
p = task.payload
expected = q.roots if level != "advanced" else select_roots(q, p.final_roots)

# --- Task -------------------------------------------------------------------
with st.container(border=True):
    st.subheader(p.title)
    if task.shown_latex:
        st.latex(task.shown_latex)
    st.markdown(p.statement)
    st.caption(
        f"✔ Math verified by SymPy"
        + (f" after {task.attempts} AI attempts" if task.attempts > 1 else "")
    )

if st.button("🔄 Generate a different version") and generate():
    st.rerun()

# --- Hints ------------------------------------------------------------------
st.markdown("#### 💡 Hints")
for i, hint in enumerate(p.hints[: progress["hints"]], start=1):
    st.markdown(f"**Hint {i}.** {hint}")
if progress["hints"] < len(p.hints):
    def _next_hint() -> None:
        progress["hints"] += 1

    st.button(f"Show hint {progress['hints'] + 1} of {len(p.hints)}", on_click=_next_hint)

# --- Answer -----------------------------------------------------------------
st.markdown("#### ✍️ Your answer")
with st.form(key=f"answer-{key}"):
    answer = st.text_input(
        "Roots separated by ';'",
        placeholder="e.g.  2; -1/3   or   (5+sqrt(13))/2   or   no roots",
    )
    submitted = st.form_submit_button("Check answer", type="primary")

if submitted and answer.strip():
    progress["attempts"] += 1
    result = check_answer(answer, expected)
    try:
        with st.spinner("Getting feedback…"):
            fb = tutor.feedback(task, answer, result.status, language)
    except Exception as e:
        fb = friendly_error(e)
    progress["result"] = {"status": result.status, "found": result.found,
                          "expected": result.expected, "feedback": fb}

if progress["result"]:
    r = progress["result"]
    if r["status"] == "correct":
        st.success("✅ Correct! (checked by SymPy)")
        if submitted and progress["attempts"] == 1:
            st.balloons()
    elif r["status"] == "partial":
        st.warning(f"🟡 Partly right: you found {r['found']} of {r['expected']} root(s), "
                   "or there is an extra value.")
    elif r["status"] == "wrong":
        st.error("❌ Not quite yet.")
    else:
        st.info("🤔 I couldn't read that answer.")
    with st.chat_message("assistant", avatar="🧑‍🏫"):
        st.markdown(r["feedback"])
    if r["status"] != "correct" and progress["attempts"] >= 2 and progress["hints"] < len(p.hints):
        st.caption("Tip: try opening the next hint above.")

# --- Solution ---------------------------------------------------------------
st.markdown("#### 📖 Solution")
if not progress["solution"]:
    def _reveal() -> None:
        progress["solution"] = True

    st.button("Show step-by-step solution", on_click=_reveal)
else:
    with st.container(border=True):
        for i, step in enumerate(p.solution_steps, start=1):
            st.markdown(f"**{i}.** {step}")
        st.markdown(f"**Answer:** {p.final_answer}")
    roots_text = ", ".join(f"${format_root(r)}$" for r in q.roots) or "none (D < 0)"
    st.caption(f"SymPy check: $D = {q.discriminant}$; real roots of ${q.latex}$: {roots_text}")

# --- Ask the tutor ----------------------------------------------------------
st.markdown("#### 💬 Ask the AI tutor")
st.caption("Ask anything about this task. The tutor answers in the style of your level.")
for message in progress["chat"]:
    with st.chat_message(message["role"], avatar="🧑‍🏫" if message["role"] == "assistant" else None):
        st.markdown(message["content"])

question = st.chat_input("e.g. Why do we need the discriminant?")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    try:
        with st.spinner("Thinking…"):
            reply = tutor.chat(task, progress["chat"], question, progress["solution"], language)
    except Exception as e:
        reply = friendly_error(e)
    progress["chat"] += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
    st.rerun()
