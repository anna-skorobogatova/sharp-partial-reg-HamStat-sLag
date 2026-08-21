#!/usr/bin/env python3
"""Independent directed-rounding validation of the Cartan-cubic certificate.

This checker deliberately does *not* import python-flint, Arb, SymPy, or any
module from the primary validator.  It uses only the Python standard library.
All interval endpoints are Decimal numbers evaluated with explicit outward
rounding.  The functions D and N and their first derivatives are evaluated
from the complex definitions of K and T, not from the expanded real
polynomials used by the Arb implementation.

The exact rational Taylor polynomials in witness_polynomials.json are treated
only as proposed witnesses.  Their residuals, path tubes, contraction bounds,
Gronwall bounds, endpoint signs, and the uniform lower bound for D are all
recomputed here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import (
    Context,
    Decimal,
    ROUND_CEILING,
    ROUND_FLOOR,
    localcontext,
)
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

PRECISION = 160
LOCAL_CELLS = 96
REGULAR_CELLS = 24

DOWN = Context(prec=PRECISION, rounding=ROUND_FLOOR)
UP = Context(prec=PRECISION, rounding=ROUND_CEILING)


def down_add(a: Decimal, b: Decimal) -> Decimal:
    return DOWN.add(a, b)


def up_add(a: Decimal, b: Decimal) -> Decimal:
    return UP.add(a, b)


def down_sub(a: Decimal, b: Decimal) -> Decimal:
    return DOWN.subtract(a, b)


def up_sub(a: Decimal, b: Decimal) -> Decimal:
    return UP.subtract(a, b)


def down_mul(a: Decimal, b: Decimal) -> Decimal:
    return DOWN.multiply(a, b)


def up_mul(a: Decimal, b: Decimal) -> Decimal:
    return UP.multiply(a, b)


def down_div(a: Decimal, b: Decimal) -> Decimal:
    return DOWN.divide(a, b)


def up_div(a: Decimal, b: Decimal) -> Decimal:
    return UP.divide(a, b)


def fraction_bounds(value: Fraction) -> tuple[Decimal, Decimal]:
    n = Decimal(value.numerator)
    d = Decimal(value.denominator)
    return down_div(n, d), up_div(n, d)


def parse_fraction(value: str | int) -> Fraction:
    return Fraction(str(value))


@dataclass(frozen=True)
class I:
    lo: Decimal
    hi: Decimal

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"reversed interval [{self.lo},{self.hi}]")

    @staticmethod
    def zero() -> "I":
        return I(Decimal(0), Decimal(0))

    @staticmethod
    def point(value: int | Decimal | Fraction | str) -> "I":
        if isinstance(value, Fraction):
            lo, hi = fraction_bounds(value)
            return I(lo, hi)
        if isinstance(value, Decimal):
            return I(value, value)
        if isinstance(value, int):
            d = Decimal(value)
            return I(d, d)
        return I.point(parse_fraction(value))

    @staticmethod
    def hull(a: Fraction | str, b: Fraction | str) -> "I":
        aa = I.point(a)
        bb = I.point(b)
        return I(min(aa.lo, bb.lo), max(aa.hi, bb.hi))

    @staticmethod
    def symmetric(radius: Decimal | Fraction | str) -> "I":
        r = I.point(radius).abs_upper()
        return I(r.copy_negate(), r)

    def __add__(self, other: object) -> "I":
        b = as_i(other)
        return I(down_add(self.lo, b.lo), up_add(self.hi, b.hi))

    __radd__ = __add__

    def __neg__(self) -> "I":
        return I(self.hi.copy_negate(), self.lo.copy_negate())

    def __sub__(self, other: object) -> "I":
        b = as_i(other)
        return I(down_sub(self.lo, b.hi), up_sub(self.hi, b.lo))

    def __rsub__(self, other: object) -> "I":
        return as_i(other) - self

    def __mul__(self, other: object) -> "I":
        b = as_i(other)
        lower_candidates = [
            down_mul(self.lo, b.lo),
            down_mul(self.lo, b.hi),
            down_mul(self.hi, b.lo),
            down_mul(self.hi, b.hi),
        ]
        upper_candidates = [
            up_mul(self.lo, b.lo),
            up_mul(self.lo, b.hi),
            up_mul(self.hi, b.lo),
            up_mul(self.hi, b.hi),
        ]
        return I(min(lower_candidates), max(upper_candidates))

    __rmul__ = __mul__

    def reciprocal(self) -> "I":
        if self.lo <= 0 <= self.hi:
            raise ZeroDivisionError(f"interval contains zero: {self}")
        if self.lo > 0:
            return I(down_div(Decimal(1), self.hi), up_div(Decimal(1), self.lo))
        return I(down_div(Decimal(1), self.hi), up_div(Decimal(1), self.lo))

    def __truediv__(self, other: object) -> "I":
        return self * as_i(other).reciprocal()

    def __rtruediv__(self, other: object) -> "I":
        return as_i(other) / self

    def __pow__(self, exponent: int) -> "I":
        if exponent < 0:
            return (self.reciprocal()) ** (-exponent)
        result = I.point(1)
        base = self
        n = exponent
        while n:
            if n & 1:
                result = result * base
            n //= 2
            if n:
                base = base * base
        return result

    def abs_upper(self) -> Decimal:
        return max(self.lo.copy_abs(), self.hi.copy_abs())

    def union(self, other: "I") -> "I":
        return I(min(self.lo, other.lo), max(self.hi, other.hi))

    def add_radius(self, radius: Decimal | Fraction | str) -> "I":
        return self + I.symmetric(radius)

    def distance_upper(self, point: Fraction | str) -> Decimal:
        return (self - I.point(point)).abs_upper()

    def contains(self, other: "I") -> bool:
        return self.lo <= other.lo and other.hi <= self.hi

    def __str__(self) -> str:
        return f"[{self.lo},{self.hi}]"


def as_i(value: object) -> I:
    if isinstance(value, I):
        return value
    if isinstance(value, (int, Decimal, Fraction, str)):
        return I.point(value)
    raise TypeError(type(value))


@dataclass(frozen=True)
class C:
    re: I
    im: I

    @staticmethod
    def zero() -> "C":
        return C(I.zero(), I.zero())

    @staticmethod
    def real(value: object) -> "C":
        return C(as_i(value), I.zero())

    def __add__(self, other: object) -> "C":
        b = as_c(other)
        return C(self.re + b.re, self.im + b.im)

    __radd__ = __add__

    def __neg__(self) -> "C":
        return C(-self.re, -self.im)

    def __sub__(self, other: object) -> "C":
        return self + (-as_c(other))

    def __rsub__(self, other: object) -> "C":
        return as_c(other) - self

    def __mul__(self, other: object) -> "C":
        b = as_c(other)
        return C(self.re * b.re - self.im * b.im, self.re * b.im + self.im * b.re)

    __rmul__ = __mul__

    def __pow__(self, exponent: int) -> "C":
        if exponent < 0:
            raise ValueError("negative complex powers are not needed")
        result = C.real(1)
        base = self
        n = exponent
        while n:
            if n & 1:
                result = result * base
            n //= 2
            if n:
                base = base * base
        return result


def as_c(value: object) -> C:
    return value if isinstance(value, C) else C.real(value)


IUNIT = C(I.zero(), I.point(1))


def dn_derivatives(s: I, f: I, y: I) -> tuple[I, I, I, I, I, I]:
    """Return D,N,D_f,D_y,N_f,N_y from the complex K,T formulas."""
    K = C(I.point(1), I.point(2) * f)
    one_minus_s2 = I.point(1) - s * s
    p2 = I.point(9) * one_minus_s2 * y * y
    A = K**2 + p2
    T = K**3 + K * (I.point(3) * p2) - IUNIT * (I.point(3) * s * y) * (K**2 * I.point(3) + p2)
    KT = K * T
    AT = A * T
    D = KT.re
    N = AT.im

    def one_derivative(dK: C, dp2: I, dy_var: I) -> tuple[I, I]:
        dT = (
            K**2 * dK * I.point(3)
            + (K * dp2 + dK * p2) * I.point(3)
            - IUNIT
            * (I.point(3) * s)
            * ((K**2 * I.point(3) + p2) * dy_var + (K * dK * I.point(6) + dp2) * y)
        )
        dD = (dK * T + K * dT).re
        dA = K * dK * I.point(2) + dp2
        dN = (dA * T + A * dT).im
        return dD, dN

    Df, Nf = one_derivative(C(I.zero(), I.point(2)), I.zero(), I.zero())
    Dy, Ny = one_derivative(C.zero(), I.point(18) * one_minus_s2 * y, I.point(1))
    return D, N, Df, Dy, Nf, Ny


def gq_derivatives(x: I, f: I, y: I, regular: bool) -> tuple[I, I, I, I]:
    s = I.point(1) - x
    D, N, Df, Dy, Nf, Ny = dn_derivatives(s, f, y)
    H = N - I.point(9) * s * y * D
    Hf = Nf - I.point(9) * s * y * Df
    Hy = Ny - I.point(9) * s * (D + y * Dy)
    factor = I.point(9) * (I.point(2) - x)
    if regular:
        factor = factor * x
    denominator = factor * D
    denominator_f = factor * Df
    denominator_y = factor * Dy
    value = H / denominator
    df = (Hf * denominator - H * denominator_f) / (denominator * denominator)
    dy = (Hy * denominator - H * denominator_y) / (denominator * denominator)
    return value, df, dy, D


# Polynomial arithmetic with interval coefficients.  Coefficients are stored
# in increasing order.
Poly = list[I]
CPoly = list[C]


def pzero() -> Poly:
    return [I.zero()]


def cpzero() -> CPoly:
    return [C.zero()]


def padd(a: Sequence[I], b: Sequence[I]) -> Poly:
    n = max(len(a), len(b))
    return [(a[k] if k < len(a) else I.zero()) + (b[k] if k < len(b) else I.zero()) for k in range(n)]


def psub(a: Sequence[I], b: Sequence[I]) -> Poly:
    n = max(len(a), len(b))
    return [(a[k] if k < len(a) else I.zero()) - (b[k] if k < len(b) else I.zero()) for k in range(n)]


def pmul(a: Sequence[I], b: Sequence[I]) -> Poly:
    out = [I.zero() for _ in range(len(a) + len(b) - 1)]
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] = out[i + j] + ai * bj
    return out


def pscale(a: Sequence[I], scalar: object) -> Poly:
    s = as_i(scalar)
    return [coefficient * s for coefficient in a]


def ppow(a: Sequence[I], exponent: int) -> Poly:
    out = [I.point(1)]
    base = list(a)
    n = exponent
    while n:
        if n & 1:
            out = pmul(out, base)
        n //= 2
        if n:
            base = pmul(base, base)
    return out


def pderivative(a: Sequence[I]) -> Poly:
    if len(a) <= 1:
        return [I.zero()]
    return [a[k] * I.point(k) for k in range(1, len(a))]


def peval(a: Sequence[I], x: I) -> I:
    out = I.zero()
    for coefficient in reversed(a):
        out = out * x + coefficient
    return out


def cpadd(a: Sequence[C], b: Sequence[C]) -> CPoly:
    n = max(len(a), len(b))
    return [(a[k] if k < len(a) else C.zero()) + (b[k] if k < len(b) else C.zero()) for k in range(n)]


def cpsub(a: Sequence[C], b: Sequence[C]) -> CPoly:
    n = max(len(a), len(b))
    return [(a[k] if k < len(a) else C.zero()) - (b[k] if k < len(b) else C.zero()) for k in range(n)]


def cpmul(a: Sequence[C], b: Sequence[C]) -> CPoly:
    out = [C.zero() for _ in range(len(a) + len(b) - 1)]
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] = out[i + j] + ai * bj
    return out


def cpscale_real(a: Sequence[C], scalar: Sequence[I]) -> CPoly:
    return cpmul(a, [C.real(value) for value in scalar])


def cppow(a: Sequence[C], exponent: int) -> CPoly:
    out = [C.real(1)]
    base = list(a)
    n = exponent
    while n:
        if n & 1:
            out = cpmul(out, base)
        n //= 2
        if n:
            base = cpmul(base, base)
    return out


def real_to_complex_poly(a: Sequence[I]) -> CPoly:
    return [C.real(value) for value in a]


def complex_scale_i(a: Sequence[C]) -> CPoly:
    return [IUNIT * value for value in a]


def polynomial_dn_complex(S: Poly, F: Poly, Y: Poly) -> tuple[Poly, Poly]:
    K = cpadd([C.real(1)], [C(I.zero(), I.point(2) * value) for value in F])
    S2 = pmul(S, S)
    Y2 = pmul(Y, Y)
    p2 = pscale(pmul(psub([I.point(1)], S2), Y2), 9)
    K2 = cpmul(K, K)
    K3 = cpmul(K2, K)
    term2 = cpscale_real(K, pscale(p2, 3))
    inner = cpadd(cpscale_real(K2, [I.point(3)]), real_to_complex_poly(p2))
    SY = pmul(S, Y)
    term3 = cpscale_real(complex_scale_i(inner), pscale(SY, -3))
    T = cpadd(cpadd(K3, term2), term3)
    KT = cpmul(K, T)
    A = cpadd(K2, real_to_complex_poly(p2))
    AT = cpmul(A, T)
    return [coefficient.re for coefficient in KT], [coefficient.im for coefficient in AT]


def residual_polynomial(F: Poly, Y: Poly, x0: Fraction, regular: bool) -> tuple[Poly, Poly]:
    if regular:
        X = [I.point(x0), I.point(1)]
        S = [I.point(1 - x0), I.point(-1)]
    else:
        X = [I.zero(), I.point(1)]
        S = [I.point(1), I.point(-1)]
    D, N = polynomial_dn_complex(S, F, Y)
    H = psub(N, pscale(pmul(pmul(S, Y), D), 9))
    if regular:
        denominator = pscale(pmul(pmul(X, psub([I.point(2)], X)), D), 9)
        numerator = psub(pmul(pderivative(Y), denominator), H)
    else:
        denominator = pscale(pmul(psub([I.point(2)], X), D), 9)
        numerator = psub(pmul(pmul(X, pderivative(Y)), denominator), H)
    return numerator, denominator


def l1_bound(poly: Sequence[I], h: Fraction) -> Decimal:
    hp = I.point(1)
    hb = I.point(h)
    total = Decimal(0)
    for coefficient in poly:
        total = up_add(total, up_mul(coefficient.abs_upper(), hp.abs_upper()))
        hp = hp * hb
    return total


def exp_upper(x: Decimal) -> Decimal:
    """Rigorous upper bound for exp(x), x>=0, by scaling and a Taylor tail."""
    if x < 0:
        raise ValueError("exp_upper is only used for nonnegative arguments")
    t = x
    squarings = 0
    eighth = Decimal(1) / Decimal(8)
    while t > eighth:
        t = up_div(t, Decimal(2))
        squarings += 1
    term = Decimal(1)
    total = Decimal(1)
    n = 32
    for k in range(1, n + 1):
        term = up_div(up_mul(term, t), Decimal(k))
        total = up_add(total, term)
    first_omitted = up_div(up_mul(term, t), Decimal(n + 1))
    ratio = up_div(t, Decimal(n + 2))
    tail = up_div(first_omitted, down_sub(Decimal(1), ratio))
    result = up_add(total, tail)
    for _ in range(squarings):
        result = up_mul(result, result)
    return result


def hash_polynomial(f_strings: Sequence[str], y_strings: Sequence[str]) -> str:
    payload = "F\n" + "\n".join(f_strings) + "\nY\n" + "\n".join(y_strings)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def interval_from_record(record: dict[str, str]) -> I:
    return I.hull(record["lower_rational"], record["upper_rational"])


def decimal_sci(value: Decimal, digits: int = 40) -> str:
    with localcontext() as ctx:
        ctx.prec = digits
        return f"{+value:.{digits-1}E}"


def local_validate(run_name: str, run: dict, witness: dict, config: dict) -> tuple[tuple[I, I], Decimal, dict]:
    local = run["local_taylor_validation"]
    f_strings = witness["local"]["f_coefficients_rational"]
    y_strings = witness["local"]["y_coefficients_rational"]
    if hash_polynomial(f_strings, y_strings) != local["polynomial_sha256"]:
        raise AssertionError(f"{run_name}: local witness hash mismatch")
    F = [I.point(value) for value in f_strings]
    Y = [I.point(value) for value in y_strings]
    h = parse_fraction(config["local_h"])
    trial = I.point(config["local_trial_radius"]).hi
    A_radius = I.point(local["A_radius"]).hi

    # Endpoint compatibility: H(1,A,y)=0 has a unique root in the archived
    # B interval because H_y<0 and the endpoint signs bracket zero.  The A
    # interval is subdivided here to avoid artificial dependency inflation.
    A_box = I.hull(local["A_lower"], local["A_upper"])
    B_box = interval_from_record(local["B_interval"])
    _, _, _, Dy_box, _, Ny_box = dn_derivatives(I.point(1), A_box, B_box)
    D_box, _N_box, _, _, _, _ = dn_derivatives(I.point(1), A_box, B_box)
    Hy_box = Ny_box - I.point(9) * (D_box + B_box * Dy_box)
    if not Hy_box.hi < 0:
        raise AssertionError(f"{run_name}: endpoint H_y is not negative")
    a_lo = parse_fraction(local["A_lower"])
    a_hi = parse_fraction(local["A_upper"])
    endpoint_cells = 2048 if a_lo != a_hi else 1
    H_lower_min: Decimal | None = None
    H_upper_max: Decimal | None = None
    for j in range(endpoint_cells):
        acell = I.hull(
            a_lo + (a_hi-a_lo)*j/endpoint_cells,
            a_lo + (a_hi-a_lo)*(j+1)/endpoint_cells,
        )
        D_lo_y, N_lo_y, *_ = dn_derivatives(I.point(1), acell, I.point(B_box.lo))
        D_hi_y, N_hi_y, *_ = dn_derivatives(I.point(1), acell, I.point(B_box.hi))
        H_lo = N_lo_y - I.point(9) * I.point(B_box.lo) * D_lo_y
        H_hi = N_hi_y - I.point(9) * I.point(B_box.hi) * D_hi_y
        H_lower_min = H_lo.lo if H_lower_min is None else min(H_lower_min, H_lo.lo)
        H_upper_max = H_hi.hi if H_upper_max is None else max(H_upper_max, H_hi.hi)
        if not (H_lo.lo > 0 and H_hi.hi < 0):
            raise AssertionError(
                f"{run_name}: endpoint B interval fails to bracket H=0 in A-cell {j}"
            )
    assert H_lower_min is not None and H_upper_max is not None

    numerator, _ = residual_polynomial(F, Y, Fraction(0), regular=False)
    residual_num = l1_bound(numerator, h)

    coarse_f_error = up_add(A_radius, up_mul(I.point(h).hi, trial))
    Mf = Decimal(0)
    My = Decimal(0)
    D_coarse_lo: Decimal | None = None
    for cell in range(LOCAL_CELLS):
        left = h * cell / LOCAL_CELLS
        right = h * (cell + 1) / LOCAL_CELLS
        t = I.hull(left, right)
        fbox = peval(F, t).add_radius(coarse_f_error)
        ybox = peval(Y, t).add_radius(trial)
        _, gf, gy, Dbox = gq_derivatives(t, fbox, ybox, regular=False)
        if Dbox.lo <= 0:
            raise AssertionError(f"{run_name}: local coarse D failed in cell {cell}")
        Mf = max(Mf, gf.abs_upper())
        My = max(My, (gy + I.point(1)).abs_upper())
        D_coarse_lo = Dbox.lo if D_coarse_lo is None else min(D_coarse_lo, Dbox.lo)
    assert D_coarse_lo is not None
    denominator_lower = down_mul(Decimal(9), down_mul(I.point(2 - h).lo, D_coarse_lo))
    Rs = up_div(residual_num, denominator_lower)
    q = up_add(up_mul(I.point(h).hi, Mf), My)
    if q >= 1:
        raise AssertionError(f"{run_name}: local contraction q={q}")
    V = up_div(up_add(up_mul(Mf, A_radius), Rs), down_sub(Decimal(1), q))
    if V >= trial:
        raise AssertionError(f"{run_name}: local tail exceeds trial radius")
    f_error = up_add(A_radius, up_mul(I.point(h).hi, V))

    D_narrow_lo: Decimal | None = None
    for cell in range(LOCAL_CELLS):
        left = h * cell / LOCAL_CELLS
        right = h * (cell + 1) / LOCAL_CELLS
        t = I.hull(left, right)
        fbox = peval(F, t).add_radius(f_error)
        ybox = peval(Y, t).add_radius(V)
        Dbox, *_ = dn_derivatives(I.point(1) - t, fbox, ybox)
        if Dbox.lo <= 0:
            raise AssertionError(f"{run_name}: local narrow D failed in cell {cell}")
        D_narrow_lo = Dbox.lo if D_narrow_lo is None else min(D_narrow_lo, Dbox.lo)
    assert D_narrow_lo is not None
    state = (peval(F, I.point(h)).add_radius(f_error), peval(Y, I.point(h)).add_radius(V))
    return state, D_narrow_lo, {
        "endpoint_Hy_upper": str(Hy_box.hi),
        "endpoint_H_at_B_lower_lower": str(H_lower_min),
        "endpoint_H_at_B_upper_upper": str(H_upper_max),
        "endpoint_A_subdivision_cells": endpoint_cells,
        "residual_numerator_upper": str(residual_num),
        "residual_denominator_lower": str(denominator_lower),
        "M_f_upper": str(Mf),
        "M_y_plus_one_upper": str(My),
        "contraction_q_upper": str(q),
        "tail_y_upper": str(V),
        "f_error_at_h_upper": str(f_error),
        "D_lower": str(D_narrow_lo),
    }


def regular_validate(run_name: str, run: dict, witness: dict, state: tuple[I, I], D_global_lo: Decimal, config: dict) -> tuple[tuple[I, I], Decimal, list[dict]]:
    records: list[dict] = []
    witness_steps = witness["steps"]
    steps = run["continuation_steps"]
    if len(witness_steps) != len(steps):
        raise AssertionError(f"{run_name}: witness step count mismatch")

    for step, polynomial in zip(steps, witness_steps, strict=True):
        index = int(step["index"])
        if polynomial["index"] != index:
            raise AssertionError(f"{run_name} step {index}: witness index mismatch")
        f_strings = polynomial["f_coefficients_rational"]
        y_strings = polynomial["y_coefficients_rational"]
        if hash_polynomial(f_strings, y_strings) != step["polynomial_sha256"]:
            raise AssertionError(f"{run_name} step {index}: polynomial hash mismatch")
        F = [I.point(value) for value in f_strings]
        Y = [I.point(value) for value in y_strings]
        x0 = parse_fraction(step["x0"])
        h = parse_fraction(step["h"])
        radius = I.point(step["trial_radius"]).hi
        f_center = parse_fraction(step["center_in_f"])
        y_center = parse_fraction(step["center_in_y"])
        if not (F[0].contains(I.point(f_center)) and Y[0].contains(I.point(y_center))):
            raise AssertionError(f"{run_name} step {index}: constant coefficients mismatch")
        E0 = max(state[0].distance_upper(f_center), state[1].distance_upper(y_center))

        numerator, _ = residual_polynomial(F, Y, x0, regular=True)
        residual_num = l1_bound(numerator, h)

        L = Decimal(0)
        D_step_lo: Decimal | None = None
        for cell in range(REGULAR_CELLS):
            left = h * cell / REGULAR_CELLS
            right = h * (cell + 1) / REGULAR_CELLS
            t = I.hull(left, right)
            xbox = I.point(x0) + t
            fbox = peval(F, t).add_radius(radius)
            ybox = peval(Y, t).add_radius(radius)
            _, qf, qy, Dbox = gq_derivatives(xbox, fbox, ybox, regular=True)
            if Dbox.lo <= 0:
                raise AssertionError(f"{run_name} step {index}: D failed in cell {cell}")
            L = max(L, up_add(Decimal(1), up_add(qf.abs_upper(), qy.abs_upper())))
            D_step_lo = Dbox.lo if D_step_lo is None else min(D_step_lo, Dbox.lo)
        assert D_step_lo is not None
        D_global_lo = min(D_global_lo, D_step_lo)
        denominator_lower = down_mul(
            Decimal(9),
            down_mul(
                I.point(x0).lo,
                down_mul(I.point(2 - (x0 + h)).lo, D_step_lo),
            ),
        )
        rho = up_div(residual_num, denominator_lower)
        Lh = up_mul(L, I.point(h).hi)
        beta = up_mul(
            up_add(E0, up_mul(rho, I.point(h).hi)),
            exp_upper(Lh),
        )
        if beta >= radius:
            raise AssertionError(
                f"{run_name} step {index}: beta={beta} exceeds radius={radius}"
            )
        pend_f = peval(F, I.point(h))
        pend_y = peval(Y, I.point(h))
        state = (pend_f.add_radius(beta), pend_y.add_radius(beta))
        records.append(
            {
                "index": index,
                "x0": str(x0),
                "h": str(h),
                "initial_error_upper": str(E0),
                "residual_numerator_upper": str(residual_num),
                "residual_denominator_lower": str(denominator_lower),
                "rho_upper": str(rho),
                "L_upper": str(L),
                "beta_upper": str(beta),
                "D_lower": str(D_step_lo),
            }
        )
        if index % 20 == 0 or index == len(steps) - 1:
            print(
                f"{run_name}: step {index:03d}, "
                f"beta<={decimal_sci(beta, 14)}, D>={decimal_sci(D_step_lo, 14)}",
                flush=True,
            )
    return state, D_global_lo, records


def main() -> int:
    here = Path(__file__).resolve().parent
    certificate = json.loads((here / "certificate.json").read_text(encoding="utf-8"))
    witness = json.loads((here / "witness_polynomials.json").read_text(encoding="utf-8"))
    config = certificate["configuration"]
    results = {
        "schema": "special-lagrangian-cartancubic-independent-decimal-validation-v1",
        "precision_decimal_digits": PRECISION,
        "interval_engine": "Python Decimal with explicit ROUND_FLOOR/ROUND_CEILING",
        "D_N_evaluation": "complex K,T formulas, not expanded real polynomials",
        "local_subdivision_cells": LOCAL_CELLS,
        "regular_subdivision_cells": REGULAR_CELLS,
        "runs": {},
    }

    for run_name in ["A_minus", "A_plus", "A_interval"]:
        print(f"validating {run_name}", flush=True)
        run = certificate["runs"][run_name]
        run_witness = witness["runs"][run_name]
        state, D_lower, local_result = local_validate(run_name, run, run_witness, config)
        state, D_lower, step_results = regular_validate(
            run_name, run, run_witness, state, D_lower, config
        )
        if run_name == "A_minus" and not state[0].lo > 0:
            raise AssertionError(f"A_minus endpoint is not certified positive: {state[0]}")
        if run_name == "A_plus" and not state[0].hi < 0:
            raise AssertionError(f"A_plus endpoint is not certified negative: {state[0]}")
        if run_name == "A_interval" and not D_lower > Decimal("4.2964512605"):
            raise AssertionError(f"uniform D lower bound is too small: {D_lower}")
        results["runs"][run_name] = {
            "local": local_result,
            "steps": step_results,
            "endpoint_f_lower": str(state[0].lo),
            "endpoint_f_upper": str(state[0].hi),
            "endpoint_y_lower": str(state[1].lo),
            "endpoint_y_upper": str(state[1].hi),
            "global_D_lower": str(D_lower),
        }
        print(
            f"PASS {run_name}: f(0) in {state[0]}, global D >= {D_lower}",
            flush=True,
        )

    destination = here / "independent_decimal_results.json"
    destination.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("INDEPENDENT DECIMAL VALIDATION PASSED", flush=True)
    print(f"wrote {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
