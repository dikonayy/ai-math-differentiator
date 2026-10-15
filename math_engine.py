"""
Deterministic math core.

The AI is only allowed to *explain* and *re-word* problems. Every number that
decides whether an answer is correct comes from this module (SymPy), so a
hallucinating model can never mark a wrong answer as right.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

x = sp.Symbol("x")

# Two answers closer than this are treated as equal, so "3.30" is accepted for (5+sqrt(13))/2.
TOLERANCE = 0.01


@dataclass(frozen=True)
class Quadratic:
    """The core problem: a*x^2 + b*x + c = 0 with integer coefficients."""

    a: int
    b: int
    c: int

    def __post_init__(self):
        if self.a == 0:
            raise ValueError("a must not be 0, otherwise the equation is linear, not quadratic.")

    @property
    def expr(self) -> sp.Expr:
        return self.a * x**2 + self.b * x + self.c

    @property
    def discriminant(self) -> int:
        return self.b**2 - 4 * self.a * self.c

    @property
    def roots(self) -> List[sp.Expr]:
        """Exact real roots sorted ascending; empty when D < 0, one root when D = 0."""
        if self.discriminant < 0:
            return []
        return sorted(sp.solve(sp.Eq(self.expr, 0), x), key=float)

    @property
    def latex(self) -> str:
        return f"{sp.latex(self.expr)} = 0"

    @property
    def plain(self) -> str:
        return f"{sp.sstr(self.expr)} = 0"


def random_quadratic(rng: Optional[random.Random] = None) -> Quadratic:
    """Random equation with rational roots, at least one of them positive.

    Built from factors (a*x - m)(x - r) so the roots are always "nice"; the
    positive root lets the Advanced level wrap it into a real-world story.
    """
    rng = rng or random.Random()
    a = rng.choice([1, 1, 2, 3])
    r = rng.randint(1, 9)
    m = rng.choice([i for i in range(-9, 10) if i != 0 and i != a * r])
    # (a*x - m)(x - r) = a*x^2 - (a*r + m)*x + m*r  ->  roots r and m/a
    return Quadratic(a, -(a * r + m), m * r)


# ---------------------------------------------------------------------------
# How the equation is *shown* on each level (the canonical equation never changes)
# ---------------------------------------------------------------------------

def presented_equation(
    q: Quadratic, level: str, rng: Optional[random.Random] = None
) -> Optional[Tuple[sp.Expr, sp.Expr]]:
    """(left side, right side) of the equation as the student sees it, or None if hidden.

    - foundation: the equation in standard form, exactly as given
    - standard:   an equivalent but rearranged and scaled form the student must normalise first
    - advanced:   hidden, because the student has to model it from a word problem
    """
    if level == "foundation":
        return q.expr, sp.Integer(0)
    if level == "standard":
        rng = rng or random.Random()
        k = rng.choice([2, 3]) if max(abs(q.a), abs(q.b), abs(q.c)) <= 12 else 1
        # k*a*x^2 + k*c = -k*b*x : same roots, but not in standard form
        left = k * q.a * x**2 + k * q.c
        right = -k * q.b * x
        if right == 0:  # b = 0 -> move c to the right instead
            left, right = k * q.a * x**2, -k * q.c
        return left, right
    return None


def presented_latex(q: Quadratic, level: str, rng: Optional[random.Random] = None) -> Optional[str]:
    sides = presented_equation(q, level, rng)
    if sides is None:
        return None
    return f"{sp.latex(sides[0])} = {sp.latex(sides[1])}"


# ---------------------------------------------------------------------------
# Parsing and checking student answers
# ---------------------------------------------------------------------------

_NO_ROOTS = re.compile(
    r"^\s*(no( real)? (roots|solutions?)|none|нет( действительных)? корней|корней нет|∅|"
    r"түбірі жоқ|empty)\s*\.?\s*$",
    re.IGNORECASE,
)
_LABEL = re.compile(r"[xх]\s*_?\s*\d*\s*=", re.IGNORECASE)  # strips "x1 =", "x_2=", Cyrillic "х"
_SEPARATORS = re.compile(r";|,|\bor\b|\band\b|\bили\b|\bи\b|\bнемесе\b|\bжәне\b", re.IGNORECASE)
_SAFE_CHARS = re.compile(r"^[0-9+\-*/().\s]*$")
_ROOT_SIGN = re.compile(r"√\s*(\d+(?:\.\d+)?)?")  # "√13" -> "sqrt(13)", "√(…)" -> "sqrt(…)"
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def parse_roots(text: str) -> Optional[List[float]]:
    """Turn a student's answer like "x1 = 2; x2 = -1/3" into floats.

    Returns [] for "no real roots" and None if anything could not be understood.
    Only digits, + - * / ( ) . and sqrt are allowed before SymPy sees the
    input, so a public web form cannot be used to run arbitrary code.
    """
    if not text or not text.strip():
        return None
    if _NO_ROOTS.match(text):
        return []
    cleaned = _LABEL.sub(" ", text).replace("−", "-")
    cleaned = _ROOT_SIGN.sub(lambda m: f"sqrt({m.group(1)})" if m.group(1) else "sqrt", cleaned)
    values: List[float] = []
    for part in _SEPARATORS.split(cleaned):
        part = part.strip()
        if not part:
            continue
        if len(part) > 40 or "**" in part or not _SAFE_CHARS.match(part.replace("sqrt", "")):
            return None
        try:
            value = parse_expr(part, transformations=_TRANSFORMS, evaluate=True)
            number = complex(sp.N(value))
        except Exception:
            return None
        if abs(number.imag) > 1e-9:
            return None
        values.append(number.real)
    return values or None


@dataclass
class CheckResult:
    status: str  # "correct" | "partial" | "wrong" | "unparsed"
    found: int = 0
    expected: int = 0
    extra: List[float] = field(default_factory=list)

    @property
    def is_correct(self) -> bool:
        return self.status == "correct"


def _close(u: float, v: float) -> bool:
    return abs(u - v) <= TOLERANCE


def check_answer(text: str, expected_roots: Sequence[sp.Expr]) -> CheckResult:
    """Compare the student's roots with the expected ones, ignoring order and duplicates."""
    student = parse_roots(text)
    if student is None:
        return CheckResult("unparsed", expected=len(expected_roots))
    expected = [float(r) for r in expected_roots]
    found = sum(1 for e in expected if any(_close(e, s) for s in student))
    extra = [s for s in student if not any(_close(e, s) for e in expected)]
    if found == len(expected) and not extra:
        status = "correct"
    elif found > 0:
        status = "partial"
    else:
        status = "wrong"
    return CheckResult(status, found, len(expected), extra)


# ---------------------------------------------------------------------------
# Verifying what the AI produced
# ---------------------------------------------------------------------------

def is_same_equation(q: Quadratic, a: float, b: float, c: float) -> bool:
    """True if a*x^2 + b*x + c = 0 is the canonical equation multiplied by some k != 0."""
    if abs(a) < 1e-12:
        return False
    k = a / q.a
    return abs(b - k * q.b) < 1e-6 * max(1, abs(b)) and abs(c - k * q.c) < 1e-6 * max(1, abs(c))


_MODEL_CHARS = re.compile(r"^[0-9x+\-*/().\s]*$")
_BAD_POWER = re.compile(r"\*\*(?!\s*[0-3](?![0-9.*]))")  # only x**0..x**3, no towers like 9**9**9


def parse_model_side(text: str) -> Optional[sp.Expr]:
    """Parse one side of an equation the AI derived from a word problem, e.g. "x*(10 - 2*x)"."""
    text = text.replace("^", "**").replace("−", "-").replace("·", "*").strip()
    if not text or len(text) > 150 or not _MODEL_CHARS.match(text) or _BAD_POWER.search(text):
        return None
    try:
        return parse_expr(text, local_dict={"x": x}, transformations=_TRANSFORMS, evaluate=True)
    except Exception:
        return None


def equation_matches(q: Quadratic, lhs: str, rhs: str) -> bool:
    """True if lhs = rhs is equivalent to the canonical equation.

    Fractions such as 120/x are cleared first, so the numerator of lhs - rhs
    must be a non-zero multiple of a*x^2 + b*x + c.
    """
    left, right = parse_model_side(lhs), parse_model_side(rhs)
    if left is None or right is None:
        return False
    numerator = sp.fraction(sp.together(sp.expand(left - right)))[0]
    try:
        poly = sp.Poly(sp.expand(numerator), x)
    except sp.PolynomialError:
        return False
    if poly.degree() != 2:
        return False
    a, b, c = (float(poly.coeff_monomial(x**k)) for k in (2, 1, 0))
    return is_same_equation(q, a, b, c)


def roots_match(claimed: Sequence[str], q: Quadratic, *, allow_subset: bool) -> bool:
    """Check the roots the AI claims against SymPy.

    allow_subset=True is used on the Advanced level, where a real-world story may
    legitimately reject a root (e.g. a negative length), but it must keep at least one.
    """
    true_roots = [float(r) for r in q.roots]
    parsed = parse_roots("; ".join(claimed)) if claimed else []
    if parsed is None:
        return False
    if not all(any(_close(p, t) for t in true_roots) for p in parsed):
        return False
    if allow_subset:
        return len(parsed) > 0
    return all(any(_close(t, p) for p in parsed) for t in true_roots)


def select_roots(q: Quadratic, claimed: Sequence[str]) -> List[sp.Expr]:
    """The exact SymPy roots that correspond to the (already verified) roots the AI kept."""
    parsed = parse_roots("; ".join(claimed)) or []
    return [r for r in q.roots if any(_close(float(r), p) for p in parsed)]


def format_root(r: sp.Expr) -> str:
    """LaTeX for a root, with a decimal approximation when it is irrational."""
    exact = sp.latex(r)
    if r.is_rational:
        return exact
    return f"{exact} \\approx {float(r):.2f}"
