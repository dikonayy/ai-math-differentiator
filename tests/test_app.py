"""UI flow tests with Gemini replaced by a fake, so they run offline and for free."""
import pytest
from streamlit.testing.v1 import AppTest

import tutor
from math_engine import Quadratic


class FakeResponse:
    def __init__(self, parsed=None, text=""):
        self.parsed = parsed
        self.text = text


@pytest.fixture
def fake_gemini(monkeypatch):
    calls = []

    def fake_generate(self, prompt, temperature, schema=None, system=None):
        calls.append(prompt)
        if schema is None:
            return FakeResponse(text="Fake tutor reply.")
        if schema is tutor.SolverPayload:
            return FakeResponse(parsed=tutor.SolverPayload(
                unknown="width, m", equation_lhs="x*(5 - x)", equation_rhs="6", answer_roots=["3"],
            ))
        q = Quadratic(1, -5, 6)
        roots = ["3"] if "Level 3" in prompt else ["2", "3"]
        payload = tutor.TaskPayload(
            title="Fake task", statement="Solve it.", hints=["h1", "h2", "h3"],
            solution_steps=["step"], final_answer="done", final_roots=roots,
            model_a=q.a, model_b=q.b, model_c=q.c,
        )
        return FakeResponse(parsed=payload)

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(tutor.Tutor, "_generate", fake_generate)
    return calls


def start(level):
    at = AppTest.from_file("app.py", default_timeout=30)
    at.session_state["equation"] = Quadratic(1, -5, 6)
    at.run()
    [r for r in at.radio if r.label == "Choose your level"][0].set_value(level).run()
    return at


def click(at, label):
    [b for b in at.button if b.label == label][0].click().run()


def submit_answer(at, text):
    at.text_input[0].input(text)
    click(at, "Check answer")


def test_missing_key_shows_setup_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    at = AppTest.from_file("app.py", default_timeout=30)
    at.secrets["GEMINI_API_KEY"] = ""  # ignore a local .streamlit/secrets.toml
    at.run()
    assert any("GEMINI_API_KEY is not set" in e.value for e in at.error)


def test_full_flow_foundation(fake_gemini):
    at = start("foundation")
    click(at, "✨ Generate task")
    assert not at.exception
    assert any(s.value == "Fake task" for s in at.subheader)
    assert "x^{2} - 5 x + 6 = 0" in at.latex[-1].value  # the verified equation, not AI text

    click(at, "Show hint 1 of 3")
    assert any("h1" in m.value for m in at.markdown)

    submit_answer(at, "2; 3")
    assert any("Correct" in s.value for s in at.success)
    assert any("Fake tutor reply." in m.value for m in at.markdown)

    click(at, "Show step-by-step solution")
    assert any("SymPy check" in c.value for c in at.caption)

    at.chat_input[0].set_value("Why?").run()
    assert any(m.value == "Why?" for m in at.markdown)


def test_wrong_answer_is_graded_by_sympy(fake_gemini):
    at = start("standard")
    click(at, "✨ Generate task")
    submit_answer(at, "1; 6")
    assert any("Not quite" in e.value for e in at.error)
    assert "WRONG" in fake_gemini[-1]  # the AI is told the verdict, it doesn't decide it


def test_advanced_accepts_only_meaningful_root(fake_gemini):
    at = start("advanced")
    click(at, "✨ Generate task")
    submit_answer(at, "3")
    assert any("Correct" in s.value for s in at.success)


def test_bad_ai_math_is_retried_then_rejected(fake_gemini, monkeypatch):
    def always_wrong(self, prompt, temperature, schema=None, system=None):
        fake_gemini.append(prompt)
        return FakeResponse(parsed=tutor.TaskPayload(
            title="t", statement="s", hints=["a", "b", "c"], solution_steps=["x"],
            final_answer="x", final_roots=["1", "6"], model_a=1, model_b=-5, model_c=6,
        ))

    monkeypatch.setattr(tutor.Tutor, "_generate", always_wrong)
    at = start("foundation")
    click(at, "✨ Generate task")
    assert len(fake_gemini) == tutor.MAX_ATTEMPTS
    assert "PREVIOUS ATTEMPT WAS REJECTED" in fake_gemini[-1]
    assert any("could not produce a mathematically verified task" in e.value for e in at.error)


def test_advanced_story_that_does_not_fit_equation_is_rejected(fake_gemini, monkeypatch):
    """Real Gemini bug: it claimed x^2 - 10x + 16 = 0 but its story gave x(10 - 2x) = 16."""
    original = tutor.Tutor._generate

    def wrong_story(self, prompt, temperature, schema=None, system=None):
        if schema is tutor.SolverPayload:
            fake_gemini.append(prompt)
            return FakeResponse(parsed=tutor.SolverPayload(
                unknown="width", equation_lhs="x*(10 - 2*x)", equation_rhs="16", answer_roots=["2"],
            ))
        return original(self, prompt, temperature, schema, system)

    monkeypatch.setattr(tutor.Tutor, "_generate", wrong_story)
    at = start("advanced")
    click(at, "✨ Generate task")
    assert any("could not produce a mathematically verified task" in e.value for e in at.error)
    assert "NOT equivalent" in fake_gemini[-2]  # the rejection was fed back to the generator
