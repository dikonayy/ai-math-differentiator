import random

import pytest
import sympy as sp

from math_engine import (
    Quadratic,
    check_answer,
    equation_matches,
    parse_model_side,
    is_same_equation,
    parse_roots,
    presented_equation,
    presented_latex,
    random_quadratic,
    roots_match,
    select_roots,
    x,
)
from prompts import LEVEL_KEYS, build_task_prompt
from tutor import SolverPayload, TaskPayload, verify_story, verify_task


def test_roots_two_one_none():
    assert Quadratic(1, -5, 6).roots == [2, 3]
    assert Quadratic(1, -4, 4).roots == [2]
    assert Quadratic(1, 0, 1).roots == []


def test_irrational_roots_are_exact():
    roots = Quadratic(1, -5, 3).roots
    assert len(roots) == 2
    for r in roots:
        assert sp.simplify(r**2 - 5 * r + 3) == 0


def test_a_zero_rejected():
    with pytest.raises(ValueError):
        Quadratic(0, 2, 1)


@pytest.mark.parametrize("seed", range(200))
def test_random_quadratic_has_nice_positive_root(seed):
    q = random_quadratic(random.Random(seed))
    roots = q.roots
    assert len(roots) == 2
    assert all(r.is_rational for r in roots)
    assert any(r > 0 for r in roots)


@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("level", ["foundation", "standard"])
def test_presented_equation_has_same_roots(seed, level):
    rng = random.Random(seed)
    q = random_quadratic(rng) if seed % 5 else Quadratic(1, 0, -4)  # include a b = 0 case
    left, right = presented_equation(q, level, rng)
    assert sorted(sp.solve(sp.Eq(left, right), x), key=float) == q.roots


def test_presentation_per_level():
    q = Quadratic(1, -5, 6)
    assert presented_latex(q, "foundation") == q.latex
    assert presented_latex(q, "advanced") is None
    shown = presented_latex(q, "standard", random.Random(1))
    assert shown != q.latex and "=" in shown


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2; 3", [2, 3]),
        ("x1 = 2, x2 = 3", [2, 3]),
        ("3 or 2", [2, 3]),
        ("-1/2; 4", [-0.5, 4]),
        ("(5+sqrt(13))/2", [(5 + 13**0.5) / 2]),
        ("2sqrt(3)", [2 * 3**0.5]),
        ("√4", [2]),
        ("(5 - √13)/2", [(5 - 13**0.5) / 2]),
        ("√(9)", [3]),
        ("х = 2 и х = 3", [2, 3]),  # Cyrillic x and Russian "и"
    ],
)
def test_parse_roots(text, expected):
    assert sorted(parse_roots(text)) == pytest.approx(sorted(expected))


@pytest.mark.parametrize("text", ["no real roots", "No roots", "нет корней", "∅"])
def test_parse_no_roots(text):
    assert parse_roots(text) == []


@pytest.mark.parametrize(
    "text",
    ["", "hello", "__import__('os')", "9**9**9", "2x", "1" * 50, "sqrt(-4)"],
)
def test_parse_rejects_unsafe_or_invalid(text):
    assert parse_roots(text) is None


def test_check_answer_statuses():
    roots = Quadratic(1, -5, 6).roots
    assert check_answer("3; 2", roots).status == "correct"
    assert check_answer("2", roots).status == "partial"
    assert check_answer("2; 3; 7", roots).status == "partial"
    assert check_answer("-2; -3", roots).status == "wrong"
    assert check_answer("abc", roots).status == "unparsed"
    assert check_answer("no roots", Quadratic(1, 0, 1).roots).status == "correct"


def test_check_answer_accepts_decimals_for_irrational_roots():
    roots = Quadratic(1, -5, 3).roots  # (5 ± sqrt(13)) / 2  ≈ 0.70, 4.30
    assert check_answer("0.70; 4.30", roots).is_correct
    assert check_answer("(5-sqrt(13))/2; (5+sqrt(13))/2", roots).is_correct
    assert not check_answer("0.7; 4.4", roots).is_correct


def test_is_same_equation():
    q = Quadratic(1, -5, 6)
    assert is_same_equation(q, 1, -5, 6)
    assert is_same_equation(q, -2, 10, -12)
    assert is_same_equation(q, 0.5, -2.5, 3)
    assert not is_same_equation(q, 1, -5, 7)
    assert not is_same_equation(q, 0, 0, 0)


def test_roots_match():
    q = Quadratic(1, 1, -6)  # roots -3 and 2
    assert roots_match(["2", "-3"], q, allow_subset=False)
    assert not roots_match(["2"], q, allow_subset=False)
    assert roots_match(["2"], q, allow_subset=True)
    assert not roots_match([], q, allow_subset=True)
    assert not roots_match(["2", "5"], q, allow_subset=True)
    assert select_roots(q, ["2"]) == [2]


def _payload(**overrides):
    data = dict(
        title="t", statement="s", hints=["a", "b", "c"], solution_steps=["1"],
        final_answer="x = 2 or 3", final_roots=["2", "3"], model_a=1, model_b=-5, model_c=6,
    )
    data.update(overrides)
    return TaskPayload(**data)


def test_verify_task_accepts_correct_payload():
    assert verify_task(_payload(), Quadratic(1, -5, 6), "foundation") == []


def test_verify_task_catches_wrong_math():
    q = Quadratic(1, -5, 6)
    assert verify_task(_payload(final_roots=["2", "4"]), q, "standard")
    assert verify_task(_payload(model_c=7), q, "advanced")
    assert verify_task(_payload(hints=["only one"]), q, "foundation")
    # Advanced may drop a root that makes no sense in the story
    assert verify_task(_payload(final_roots=["3"]), q, "advanced") == []
    assert verify_task(_payload(final_roots=["3"]), q, "standard")


@pytest.mark.parametrize("level", LEVEL_KEYS)
def test_prompts_contain_ground_truth_and_profile(level):
    q = Quadratic(2, -3, -2)
    prompt = build_task_prompt(q, level, presented_latex(q, level, random.Random(0)), "Russian")
    assert "D = 25" in prompt
    assert "Russian" in prompt
    assert "LEARNER PROFILE" in prompt


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        ("x^2 - 10*x + 16", "0"),
        ("x*(x - 10)", "-16"),
        ("2*x^2 + 32", "20*x"),
        ("x*(10 - x)", "16"),
        ("16/x + x", "10"),  # fractions are cleared
    ],
)
def test_equation_matches_equivalent_forms(lhs, rhs):
    assert equation_matches(Quadratic(1, -10, 16), lhs, rhs)


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        ("x*(10 - 2*x)", "16"),  # the garden-bed story Gemini really produced: x^2 - 5x + 8
        ("x - 10", "16"),
        ("x^3", "1"),
        ("9**9**9", "1"),
        ("__import__('os')", "0"),
        ("y*2", "1"),
    ],
)
def test_equation_matches_rejects_wrong_or_unsafe(lhs, rhs):
    assert not equation_matches(Quadratic(1, -10, 16), lhs, rhs)


def test_parse_model_side_rejects_power_towers():
    assert parse_model_side("x**2") is not None
    assert parse_model_side("x**22") is None
    assert parse_model_side("2**9**9") is None


def test_verify_story():
    q = Quadratic(1, -10, 16)
    good = SolverPayload(unknown="w", equation_lhs="x*(10 - x)", equation_rhs="16", answer_roots=["2"])
    assert verify_story(good, q, ["2"]) == []
    assert verify_story(good, q, ["8"])  # solver and author disagree on which root answers it
    bad = SolverPayload(unknown="w", equation_lhs="x*(10 - 2*x)", equation_rhs="16", answer_roots=["2"])
    assert "NOT equivalent" in verify_story(bad, q, ["2"])[0]
