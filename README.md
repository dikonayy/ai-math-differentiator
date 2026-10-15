# 🧮 AI Math Differentiator

A web app that takes **one core math problem, solving a quadratic equation
$ax^2 + bx + c = 0$**, and uses AI (Google Gemini) to turn it into **three
difficulty levels** with adapted tasks, hints, feedback and solutions.

> Course: Educational Technology & AI · Topic 8: AI for Differentiated Learning

**Live demo:** `https://<your-app>.streamlit.app` *(replace after deploying)*

## Features

| | Level 1: Foundation | Level 2: Standard | Level 3: Advanced |
|---|---|---|---|
| **Equation shown** | Standard form, as given | Equivalent but rearranged and scaled (e.g. $2x^2 + 12 = 10x$) | Hidden: the student builds it from a word problem |
| **Task** | Guided worksheet of numbered mini-steps | Textbook "solve the equation" | Real-world modelling + interpreting roots + extension question |
| **Hints** | Direct, with the numbers substituted | Conceptual nudges, no numbers | Socratic questions |
| **Feedback** | Points at the exact mistake | Names the stage that went wrong | One guiding question |
| **Solution** | Every arithmetic step + common mistake | Concise textbook solution + alternative method | Model → solve → interpret → check |

Also:
- **Random or custom equation**: pick a random one or type your own $a, b, c$.
- **Progressive hints**: 3 per task, revealed one at a time.
- **Instant answer checking** with level-specific AI feedback.
- **"Ask the AI tutor" chat** that answers in the style of the chosen level and
  does not give away the answer until the solution is opened.
- **Output language**: English, Russian or Kazakh.

## Mathematical correctness

LLMs make arithmetic mistakes, so the AI is **never trusted with the math**:

1. **SymPy** computes the discriminant and exact roots first. They are sent to
   the model as *ground truth* inside the prompt.
2. The model returns structured JSON (`final_roots`, and the coefficients of the
   equation its task leads to). `tutor.verify_task()` checks both against SymPy.
   If anything is wrong, the error is fed back to the model and it retries
   (up to 3 attempts). An unverified task is never shown.
3. **Advanced level, independent check:** a word problem can hide a mistake
   that the model's own JSON doesn't show (in testing, Gemini once wrote a story
   that actually led to $x(10-2x)=16$ while claiming $x^2-10x+16=0$). So a second,
   separate AI call gets **only the story text, without the ground truth**, and
   derives the equation itself. SymPy then checks that this equation is
   equivalent to the canonical one. If it isn't, the task is regenerated.
4. Student answers are graded by **SymPy**, not by the AI. The AI only writes
   the feedback text. Fractions, `sqrt(...)` and decimal approximations are
   accepted (tolerance 0.01).

## How it works

```
 ┌──────────────┐  equation + level   ┌──────────────┐  prompt (ground truth +  ┌──────────┐
 │ Streamlit UI │ ──────────────────▶ │ math_engine  │  learner profile)        │  Gemini  │
 │   app.py     │                     │   (SymPy)    │ ───────────────────────▶ │   API    │
 │              │ ◀── verified task ─ │   tutor.py   │ ◀── JSON task ────────── │          │
 └──────────────┘                     └──────────────┘    verify → retry        └──────────┘
```

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface: level switcher, task, hints, answer checking, solution, chat |
| `prompts.py` | **All AI prompts**: system prompt, one learner profile per level, independent solver prompt |
| `tutor.py` | Gemini client, structured JSON output, verification (incl. independent solver) and retry loop |
| `math_engine.py` | SymPy core: roots, per-level presentation, safe answer parsing and checking |
| `tests/` | Unit tests for the math core and the AI-output verification |

## Run locally

Requires Python 3.9+ and a free Gemini API key from
[Google AI Studio](https://aistudio.google.com/apikey).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your key into it
streamlit run app.py
```

Run the tests:

```bash
pip install pytest
pytest
```

## Deploy (Streamlit Community Cloud)

1. Push this repository to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io), then **Create app** and choose
   this repo, branch `main` and file `app.py`.
3. Under **Advanced settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
4. Deploy. The key lives only in Streamlit's secret storage. It is never in
   the code or in the browser.

Optional: set `GEMINI_MODEL` in secrets to a comma-separated list of Gemini models.
By default the app tries `gemini-flash-latest`, `gemini-3.5-flash`, `gemini-flash-lite-latest`
and `gemini-3.1-flash-lite` in order. Free-tier models are often overloaded or out of quota,
and each one has its own quota, so the app switches to the next one instead of failing.

## API key security

- The key is read server-side from `st.secrets` or an environment variable.
- `.streamlit/secrets.toml` is in `.gitignore`.
- The user's answer is filtered down to digits, operators and `sqrt` before
  SymPy parses it, so the public input box cannot execute code.
