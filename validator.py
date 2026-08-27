#!/usr/bin/env python3
"""Interval-arithmetic validator for the Cartan-cubic shooting problem.

The calculation follows the two steps described in the paper.

1. On the initial interval 0 <= t <= h_0, with t = 1-s, a rational
   Taylor polynomial is controlled by the fixed-point estimate

       V = (M_f A_r + rho_0) / (1-q),    q = h_0 M_f + M_y < 1.

2. On each standard step, R_cur is the incoming error, rho_step bounds
   the polynomial residual, and L bounds the Jacobian.  Gronwall gives

       R_next = exp(L h) (R_cur + rho_step h).

The step is accepted only when R_next lies inside the trial radius and
D stays positive on the whole tube.  All inequalities use python-flint/Arb
ball arithmetic; decimal inputs and polynomial centers are exact rationals.

Running this file with no flag performs the three shooting runs and prints
only the requested progress lines and final bounds.  Running it with
--output-steps prints the complete JSON step data to standard output.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any, Callable, Iterable, Sequence

from flint import arb, arb_poly, ctx, fmpq


A_0_DECIMAL = "0.669229906092044028341339310"
A_MINUS_DECIMAL = "0.669229906092044028341339300"
A_PLUS_DECIMAL = "0.669229906092044028341339320"


@dataclass(frozen=True)
class ValidationConfig:
    precision_bits: int = 512
    initial_h: str = "0.01"
    initial_order: int = 40
    standard_order: int = 30
    initial_cells: int = 64
    standard_cells: int = 16
    initial_trial_radius: str = "1e-12"
    standard_trial_radius: str = "1e-10"
    max_standard_step: str = "0.01"
    state_center_digits: int = 115
    coefficient_center_digits: int = 130
    endpoint_slope_center_digits: int = 135
    serialization_digits: int = 180

    def apply(self) -> None:
        ctx.prec = self.precision_bits
        ctx.threads = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "precision_bits": self.precision_bits,
            "initial_h": self.initial_h,
            "initial_order": self.initial_order,
            "standard_order": self.standard_order,
            "initial_cells": self.initial_cells,
            "standard_cells": self.standard_cells,
            "initial_trial_radius": self.initial_trial_radius,
            "standard_trial_radius": self.standard_trial_radius,
            "max_standard_step": self.max_standard_step,
            "standard_step_rule": "h=min(t/4,max_standard_step,1-t)",
            "state_center_digits": self.state_center_digits,
            "coefficient_center_digits": self.coefficient_center_digits,
            "endpoint_slope_center_digits": self.endpoint_slope_center_digits,
            "serialization_digits": self.serialization_digits,
        }


DEFAULT_CONFIG = ValidationConfig()
DEFAULT_CONFIG.apply()


# ---------------------------------------------------------------------------
# Exact rational and serialized-ball helpers
# ---------------------------------------------------------------------------


def decimal_to_fmpq(value: str | Decimal) -> fmpq:
    """Convert a finite decimal (including scientific notation) exactly."""
    fraction = Fraction(Decimal(str(value)))
    return fmpq(fraction.numerator, fraction.denominator)


def rational_to_str(value: fmpq) -> str:
    return str(value)


def rational_from_scaled_integer(integer: int, exponent10: int) -> fmpq:
    if exponent10 >= 0:
        return fmpq(integer * (10**exponent10), 1)
    return fmpq(integer, 10 ** (-exponent10))


def rational_midpoint(value: arb, digits: int) -> fmpq:
    """Choose a reproducible exact decimal center from an Arb midpoint.

    Unlike ``arb.str(..., radius=False)``, mid_rad_10exp retains the requested
    midpoint digits even when the input ball itself has a wider radius.
    The difference between this center and the whole input ball is accounted
    for explicitly in the incoming-error bound.
    """
    midpoint, _radius, exponent10 = value.mid_rad_10exp(digits)
    return rational_from_scaled_integer(int(midpoint), int(exponent10))


def interval_hull(lower: fmpq | int, upper: fmpq | int) -> arb:
    return arb(lower).union(arb(upper))


def symmetric_ball(radius: arb | fmpq | int) -> arb:
    if isinstance(radius, arb):
        radius = radius.abs_upper()
    return arb(0, radius)


def ball_record(value: arb, digits: int = 180) -> dict[str, Any]:
    """Return a machine-readable outward decimal enclosure.

    The triple means

        value subset [mid_integer +/- radius_integer] * 10**exponent10.

    The exact rational endpoints are included for checkers that do not use
    Arb.
    """
    midpoint, radius, exponent10 = value.mid_rad_10exp(digits)
    midpoint_i = int(midpoint)
    radius_i = int(radius)
    exponent_i = int(exponent10)
    lower_q = rational_from_scaled_integer(midpoint_i - radius_i, exponent_i)
    upper_q = rational_from_scaled_integer(midpoint_i + radius_i, exponent_i)
    return {
        "mid_integer": str(midpoint_i),
        "radius_integer": str(radius_i),
        "exponent10": exponent_i,
        "lower_rational": str(lower_q),
        "upper_rational": str(upper_q),
        "display": value.str(50),
    }


def ball_from_record(record: dict[str, Any]) -> arb:
    return interval_hull(fmpq(record["lower_rational"]), fmpq(record["upper_rational"]))


def positive_upper(value: arb) -> arb:
    return value.abs_upper()


def max_positive(values: Iterable[arb]) -> arb:
    result = arb(0)
    for value in values:
        result = result.union(value).abs_upper()
    return result


# ---------------------------------------------------------------------------
# Truncated power-series arithmetic with Arb coefficients
# ---------------------------------------------------------------------------

Series = list[arb]


def series_constant(value: arb | fmpq | int, order: int) -> Series:
    return [arb(value)] + [arb(0)] * order


def series_add(a: Sequence[arb], b: Sequence[arb], order: int) -> Series:
    return [
        (a[index] if index < len(a) else 0)
        + (b[index] if index < len(b) else 0)
        for index in range(order + 1)
    ]


def series_subtract(a: Sequence[arb], b: Sequence[arb], order: int) -> Series:
    return [
        (a[index] if index < len(a) else 0)
        - (b[index] if index < len(b) else 0)
        for index in range(order + 1)
    ]


def series_scale(a: Sequence[arb], scalar: arb | fmpq | int, order: int) -> Series:
    return [
        (a[index] if index < len(a) else 0) * scalar
        for index in range(order + 1)
    ]


def series_multiply(a: Sequence[arb], b: Sequence[arb], order: int) -> Series:
    output = [arb(0)] * (order + 1)
    for i in range(min(len(a), order + 1)):
        for j in range(min(len(b), order + 1 - i)):
            output[i + j] += a[i] * b[j]
    return output


def series_power(a: Sequence[arb], exponent: int, order: int) -> Series:
    output = series_constant(1, order)
    base = list(a[: order + 1]) + [arb(0)] * max(0, order + 1 - len(a))
    remaining = exponent
    while remaining:
        if remaining & 1:
            output = series_multiply(output, base, order)
        remaining //= 2
        if remaining:
            base = series_multiply(base, base, order)
    return output


def series_inverse(a: Sequence[arb], order: int) -> Series:
    if not a or a[0].contains(0):
        raise ZeroDivisionError("series has a constant coefficient containing zero")
    output = [arb(0)] * (order + 1)
    output[0] = 1 / a[0]
    for k in range(1, order + 1):
        output[k] = -sum(
            (a[j] if j < len(a) else 0) * output[k - j]
            for j in range(1, k + 1)
        ) / a[0]
    return output


def series_divide(a: Sequence[arb], b: Sequence[arb], order: int) -> Series:
    return series_multiply(a, series_inverse(b, order), order)


def dn_series(s: Series, f: Series, y: Series, order: int) -> tuple[Series, Series]:
    f_powers = [None] + [series_power(f, power, order) for power in range(1, 6)]
    s_powers = [None] + [series_power(s, power, order) for power in range(1, 6)]
    y_powers = [None] + [series_power(y, power, order) for power in range(1, 6)]

    def multiply2(a: Series, b: Series) -> Series:
        return series_multiply(a, b, order)

    def multiply3(a: Series, b: Series, c: Series) -> Series:
        return series_multiply(series_multiply(a, b, order), c, order)

    D = series_constant(1, order)
    d_terms = [
        (16, f_powers[4]),
        (-72, multiply3(f_powers[3], s_powers[1], y_powers[1])),
        (108, multiply3(f_powers[2], s_powers[2], y_powers[2])),
        (-108, multiply2(f_powers[2], y_powers[2])),
        (-24, f_powers[2]),
        (-54, multiply3(f_powers[1], s_powers[3], y_powers[3])),
        (54, multiply3(f_powers[1], s_powers[1], y_powers[3])),
        (54, multiply3(f_powers[1], s_powers[1], y_powers[1])),
        (-27, multiply2(s_powers[2], y_powers[2])),
        (27, y_powers[2]),
    ]
    for coefficient, term in d_terms:
        D = series_add(D, series_scale(term, coefficient, order), order)

    N = series_constant(0, order)
    n_terms = [
        (32, f_powers[5]),
        (-144, multiply3(f_powers[4], s_powers[1], y_powers[1])),
        (288, multiply3(f_powers[3], s_powers[2], y_powers[2])),
        (-288, multiply2(f_powers[3], y_powers[2])),
        (-80, f_powers[3]),
        (-432, multiply3(f_powers[2], s_powers[3], y_powers[3])),
        (432, multiply3(f_powers[2], s_powers[1], y_powers[3])),
        (216, multiply3(f_powers[2], s_powers[1], y_powers[1])),
        (486, multiply3(f_powers[1], s_powers[4], y_powers[4])),
        (-972, multiply3(f_powers[1], s_powers[2], y_powers[4])),
        (-216, multiply3(f_powers[1], s_powers[2], y_powers[2])),
        (486, multiply2(f_powers[1], y_powers[4])),
        (216, multiply2(f_powers[1], y_powers[2])),
        (10, f_powers[1]),
        (-243, multiply2(s_powers[5], y_powers[5])),
        (486, multiply2(s_powers[3], y_powers[5])),
        (108, multiply2(s_powers[3], y_powers[3])),
        (-243, multiply2(s_powers[1], y_powers[5])),
        (-108, multiply2(s_powers[1], y_powers[3])),
        (-9, multiply2(s_powers[1], y_powers[1])),
    ]
    for coefficient, term in n_terms:
        N = series_add(N, series_scale(term, coefficient, order), order)
    return D, N


# ---------------------------------------------------------------------------
# D, N, G, Q and automatic first derivatives in (f,y)
# ---------------------------------------------------------------------------


class Dual:
    __slots__ = ("value", "df", "dy")

    def __init__(self, value: Any, df: Any = 0, dy: Any = 0) -> None:
        self.value = value
        self.df = df
        self.dy = dy

    def __add__(self, other: Any) -> "Dual":
        other = as_dual(other)
        return Dual(self.value + other.value, self.df + other.df, self.dy + other.dy)

    __radd__ = __add__

    def __neg__(self) -> "Dual":
        return Dual(-self.value, -self.df, -self.dy)

    def __sub__(self, other: Any) -> "Dual":
        return self + (-as_dual(other))

    def __rsub__(self, other: Any) -> "Dual":
        return as_dual(other) - self

    def __mul__(self, other: Any) -> "Dual":
        other = as_dual(other)
        return Dual(
            self.value * other.value,
            self.df * other.value + self.value * other.df,
            self.dy * other.value + self.value * other.dy,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "Dual":
        other = as_dual(other)
        denominator = other.value * other.value
        return Dual(
            self.value / other.value,
            (self.df * other.value - self.value * other.df) / denominator,
            (self.dy * other.value - self.value * other.dy) / denominator,
        )

    def __rtruediv__(self, other: Any) -> "Dual":
        return as_dual(other) / self

    def __pow__(self, exponent: int) -> "Dual":
        if exponent == 0:
            return Dual(1)
        output = Dual(1)
        base = self
        remaining = exponent
        while remaining:
            if remaining & 1:
                output = output * base
            remaining //= 2
            if remaining:
                base = base * base
        return output


def as_dual(value: Any) -> Dual:
    return value if isinstance(value, Dual) else Dual(value)


def dn_value(s: Any, f: Any, y: Any) -> tuple[Any, Any]:
    D = (
        16 * f**4
        - 72 * f**3 * s * y
        + 108 * f**2 * s**2 * y**2
        - 108 * f**2 * y**2
        - 24 * f**2
        - 54 * f * s**3 * y**3
        + 54 * f * s * y**3
        + 54 * f * s * y
        - 27 * s**2 * y**2
        + 27 * y**2
        + 1
    )
    N = (
        32 * f**5
        - 144 * f**4 * s * y
        + 288 * f**3 * s**2 * y**2
        - 288 * f**3 * y**2
        - 80 * f**3
        - 432 * f**2 * s**3 * y**3
        + 432 * f**2 * s * y**3
        + 216 * f**2 * s * y
        + 486 * f * s**4 * y**4
        - 972 * f * s**2 * y**4
        - 216 * f * s**2 * y**2
        + 486 * f * y**4
        + 216 * f * y**2
        + 10 * f
        - 243 * s**5 * y**5
        + 486 * s**3 * y**5
        + 108 * s**3 * y**3
        - 243 * s * y**5
        - 108 * s * y**3
        - 9 * s * y
    )
    return D, N


def initial_rhs_G(t: Any, f: Any, y: Any) -> Any:
    s = 1 - t
    D, N = dn_value(s, f, y)
    return (N - 9 * s * y * D) / (9 * (2 - t) * D)


def standard_rhs_y(t: Any, f: Any, y: Any) -> Any:
    s = 1 - t
    D, N = dn_value(s, f, y)
    return (N - 9 * s * y * D) / (9 * t * (2 - t) * D)


def endpoint_slope_B(A: arb) -> arb:
    return (2 * A + ((arb(3) / 2) * (2 * A).atan()).tan()) / 9


# ---------------------------------------------------------------------------
# Rational polynomial construction
# ---------------------------------------------------------------------------


def initial_taylor_polynomial(
    A_center: fmpq,
    y0_center: fmpq,
    order: int,
    coefficient_digits: int,
) -> tuple[list[fmpq], list[fmpq], arb]:
    """Construct the rational Taylor polynomial for the initial step.

    The temporary Arb coefficients solve the formal recurrence.  They are then
    rounded to exact decimal rationals.  The final proof does not trust the
    recurrence: it bounds the residual of the resulting rational polynomial.
    """
    f_coefficients = [arb(0)] * (order + 2)
    y_coefficients = [arb(0)] * (order + 1)
    f_coefficients[0] = arb(A_center)
    y_coefficients[0] = arb(y0_center)

    endpoint_dual = initial_rhs_G(
        arb(0),
        Dual(arb(A_center), 1, 0),
        Dual(arb(y0_center), 0, 1),
    )
    base_G_y = endpoint_dual.dy

    for k in range(1, order + 1):
        f_coefficients[k] = -y_coefficients[k - 1] / k
        s_series = [arb(1), arb(-1)] + [arb(0)] * (k - 1)
        D, N = dn_series(
            s_series,
            f_coefficients[: k + 1],
            y_coefficients[: k + 1],
            k,
        )
        H = series_subtract(
            N,
            series_scale(
                series_multiply(
                    series_multiply(s_series, y_coefficients[: k + 1], k),
                    D,
                    k,
                ),
                9,
                k,
            ),
            k,
        )
        denominator = series_scale(
            series_multiply([arb(2), arb(-1)], D, k), 9, k
        )
        G = series_divide(H, denominator, k)
        y_coefficients[k] = G[k] / (k - base_G_y)

    rational_y = [
        rational_midpoint(coefficient, coefficient_digits)
        for coefficient in y_coefficients
    ]
    rational_y[0] = y0_center
    rational_f = [A_center] + [
        -rational_y[index] / (index + 1) for index in range(order + 1)
    ]
    return rational_f, rational_y, base_G_y


def standard_taylor_polynomial(
    t0: fmpq,
    f0: fmpq,
    y0: fmpq,
    order: int,
    coefficient_digits: int,
) -> tuple[list[fmpq], list[fmpq]]:
    f_coefficients = [arb(0)] * (order + 2)
    y_coefficients = [arb(0)] * (order + 1)
    f_coefficients[0] = arb(f0)
    y_coefficients[0] = arb(y0)

    for k in range(order):
        f_coefficients[k + 1] = -y_coefficients[k] / (k + 1)
        t_series = [arb(t0), arb(1)] + [arb(0)] * max(0, k - 1)
        s_series = [arb(1 - t0), arb(-1)] + [arb(0)] * max(0, k - 1)
        D, N = dn_series(
            s_series,
            f_coefficients[: k + 1],
            y_coefficients[: k + 1],
            k,
        )
        H = series_subtract(
            N,
            series_scale(
                series_multiply(
                    series_multiply(s_series, y_coefficients[: k + 1], k),
                    D,
                    k,
                ),
                9,
                k,
            ),
            k,
        )
        denominator = series_scale(
            series_multiply(
                series_multiply(
                    t_series,
                    [arb(2 - t0), arb(-1)],
                    k,
                ),
                D,
                k,
            ),
            9,
            k,
        )
        rhs_series = series_divide(H, denominator, k)
        y_coefficients[k + 1] = rhs_series[k] / (k + 1)

    rational_y = [
        rational_midpoint(coefficient, coefficient_digits)
        for coefficient in y_coefficients
    ]
    rational_y[0] = y0
    rational_f = [f0] + [
        -rational_y[index] / (index + 1) for index in range(order + 1)
    ]
    return rational_f, rational_y


def polynomial_dn(S: arb_poly, F: arb_poly, Y: arb_poly) -> tuple[arb_poly, arb_poly]:
    D = (
        16 * F**4
        - 72 * F**3 * S * Y
        + 108 * F**2 * S**2 * Y**2
        - 108 * F**2 * Y**2
        - 24 * F**2
        - 54 * F * S**3 * Y**3
        + 54 * F * S * Y**3
        + 54 * F * S * Y
        - 27 * S**2 * Y**2
        + 27 * Y**2
        + 1
    )
    N = (
        32 * F**5
        - 144 * F**4 * S * Y
        + 288 * F**3 * S**2 * Y**2
        - 288 * F**3 * Y**2
        - 80 * F**3
        - 432 * F**2 * S**3 * Y**3
        + 432 * F**2 * S * Y**3
        + 216 * F**2 * S * Y
        + 486 * F * S**4 * Y**4
        - 972 * F * S**2 * Y**4
        - 216 * F * S**2 * Y**2
        + 486 * F * Y**4
        + 216 * F * Y**2
        + 10 * F
        - 243 * S**5 * Y**5
        + 486 * S**3 * Y**5
        + 108 * S**3 * Y**3
        - 243 * S * Y**5
        - 108 * S * Y**3
        - 9 * S * Y
    )
    return D, N


def polynomial_l1_bound(polynomial: arb_poly, h: fmpq) -> arb:
    output = arb(0)
    h_power = arb(1)
    h_ball = arb(h)
    for coefficient in polynomial.coeffs():
        output += abs(coefficient) * h_power
        h_power *= h_ball
    return output.abs_upper()


# ---------------------------------------------------------------------------
# Subdivided path bounds
# ---------------------------------------------------------------------------


def initial_path_bounds(
    F: arb_poly,
    Y: arb_poly,
    h0: fmpq,
    f_error: arb,
    y_error: arb,
    cells: int,
) -> tuple[arb, arb, arb, arb, arb]:
    M_f = arb(0)
    M_y = arb(0)
    D_all: arb | None = None
    f_all: arb | None = None
    y_all: arb | None = None

    for cell in range(cells):
        left = h0 * cell / cells
        right = h0 * (cell + 1) / cells
        t_interval = interval_hull(left, right)
        s_interval = 1 - t_interval
        f_box = F(t_interval) + symmetric_ball(f_error)
        y_box = Y(t_interval) + symmetric_ball(y_error)
        D_cell, _ = dn_value(s_interval, f_box, y_box)
        if not D_cell > 0:
            raise ArithmeticError(
                f"initial D interval contains a nonpositive value in cell {cell}: {D_cell}"
            )

        derivative = initial_rhs_G(
            t_interval,
            Dual(f_box, 1, 0),
            Dual(y_box, 0, 1),
        )
        M_f = M_f.union(abs(derivative.df).abs_upper()).abs_upper()
        M_y = M_y.union(abs(derivative.dy + 1).abs_upper()).abs_upper()
        D_all = D_cell if D_all is None else D_all.union(D_cell)
        f_all = f_box if f_all is None else f_all.union(f_box)
        y_all = y_box if y_all is None else y_all.union(y_box)

    assert D_all is not None and f_all is not None and y_all is not None
    return M_f, M_y, D_all, f_all, y_all


def standard_path_bounds(
    F: arb_poly,
    Y: arb_poly,
    t0: fmpq,
    h: fmpq,
    radius: arb,
    cells: int,
) -> tuple[arb, arb, arb, arb]:
    L_all = arb(0)
    D_all: arb | None = None
    f_all: arb | None = None
    y_all: arb | None = None

    for cell in range(cells):
        left = h * cell / cells
        right = h * (cell + 1) / cells
        tau_interval = interval_hull(left, right)
        t_interval = arb(t0) + tau_interval
        s_interval = 1 - t_interval
        f_box = F(tau_interval) + symmetric_ball(radius)
        y_box = Y(tau_interval) + symmetric_ball(radius)
        D_cell, _ = dn_value(s_interval, f_box, y_box)
        if not D_cell > 0:
            raise ArithmeticError(
                f"standard D interval contains a nonpositive value at t={t0}, "
                f"cell={cell}: {D_cell}"
            )

        derivative = standard_rhs_y(
            t_interval,
            Dual(f_box, 1, 0),
            Dual(y_box, 0, 1),
        )
        # The infinity-norm Jacobian is at most 1+|Q_f|+|Q_y|.
        L_cell = (1 + abs(derivative.df) + abs(derivative.dy)).abs_upper()
        L_all = L_all.union(L_cell).abs_upper()
        D_all = D_cell if D_all is None else D_all.union(D_cell)
        f_all = f_box if f_all is None else f_all.union(f_box)
        y_all = y_box if y_all is None else y_all.union(y_box)

    assert D_all is not None and f_all is not None and y_all is not None
    return L_all, D_all, f_all, y_all


# ---------------------------------------------------------------------------
# Initial step and standard continuation
# ---------------------------------------------------------------------------


def validate_initial_step(
    A_lower: fmpq,
    A_upper: fmpq,
    config: ValidationConfig,
) -> tuple[tuple[arb, arb], arb, dict[str, Any]]:
    h0 = decimal_to_fmpq(config.initial_h)
    A_mid = (A_lower + A_upper) / 2
    A_radius = (A_upper - A_lower) / 2
    A_interval = interval_hull(A_lower, A_upper)
    B_interval = endpoint_slope_B(A_interval)
    B_mid_ball = endpoint_slope_B(arb(A_mid))
    y0_center = rational_midpoint(
        B_mid_ball, config.endpoint_slope_center_digits
    )

    rational_f, rational_y, base_G_y = initial_taylor_polynomial(
        A_mid,
        y0_center,
        config.initial_order,
        config.coefficient_center_digits,
    )
    F = arb_poly([arb(coefficient) for coefficient in rational_f])
    Y = arb_poly([arb(coefficient) for coefficient in rational_y])

    T_polynomial = arb_poly([0, 1])
    S_polynomial = arb_poly([1, -1])
    D_polynomial, N_polynomial = polynomial_dn(S_polynomial, F, Y)
    H_polynomial = N_polynomial - 9 * S_polynomial * Y * D_polynomial
    denominator_polynomial = 9 * arb_poly([2, -1]) * D_polynomial
    residual_numerator = (
        T_polynomial * Y.derivative() * denominator_polynomial - H_polynomial
    )

    trial_radius = arb(decimal_to_fmpq(config.initial_trial_radius))
    coarse_f_error = (arb(A_radius) + arb(h0) * trial_radius).abs_upper()
    M_f, M_y, D_coarse, _coarse_f, _coarse_y = initial_path_bounds(
        F,
        Y,
        h0,
        coarse_f_error,
        trial_radius,
        config.initial_cells,
    )
    q = (arb(h0) * M_f + M_y).abs_upper()
    if not q < 1:
        raise ArithmeticError(f"initial contraction constant is not below one: {q}")

    denominator_lower = 9 * (2 - arb(h0)) * D_coarse.lower()
    if not denominator_lower > 0:
        raise ArithmeticError(
            f"initial residual denominator is not positive: {denominator_lower}"
        )
    residual_numerator_bound = polynomial_l1_bound(residual_numerator, h0)
    rho_0 = (residual_numerator_bound / denominator_lower).abs_upper()

    V = ((M_f * arb(A_radius) + rho_0) / (1 - q)).abs_upper()
    if not trial_radius > V:
        raise ArithmeticError(
            f"initial tail {V} does not fit in trial radius {trial_radius}"
        )
    f_error = (arb(A_radius) + arb(h0) * V).abs_upper()

    y0_box = arb(y0_center) + symmetric_ball(V)
    if not y0_box.contains(B_interval):
        raise ArithmeticError(
            "the initial y-box does not contain the explicit B(A) interval"
        )
    if not M_y < 1:
        raise ArithmeticError("G_y is not negative on the initial tube")

    _Mf_narrow, _My_narrow, D_narrow, f_tube, y_tube = initial_path_bounds(
        F,
        Y,
        h0,
        f_error,
        V,
        config.initial_cells,
    )

    state_out = (
        F(arb(h0)) + symmetric_ball(f_error),
        Y(arb(h0)) + symmetric_ball(V),
    )

    record = {
        "t_interval": ["0", str(h0)],
        "A_lower": str(A_lower),
        "A_upper": str(A_upper),
        "A_mid": str(A_mid),
        "A_radius": str(A_radius),
        "B_interval": ball_record(B_interval, config.serialization_digits),
        "B_polynomial_center": str(y0_center),
        "order_y": config.initial_order,
        "order_f": config.initial_order + 1,
        "subdivision_cells": config.initial_cells,
        "trial_radius": config.initial_trial_radius,
        "base_G_y": ball_record(base_G_y, config.serialization_digits),
        "f_coefficients_rational": [str(item) for item in rational_f],
        "y_coefficients_rational": [str(item) for item in rational_y],
        "residual_numerator_bound": ball_record(
            residual_numerator_bound, config.serialization_digits
        ),
        "residual_denominator_lower": ball_record(
            denominator_lower, config.serialization_digits
        ),
        "rho_0": ball_record(rho_0, config.serialization_digits),
        "M_f": ball_record(M_f, config.serialization_digits),
        "M_y": ball_record(M_y, config.serialization_digits),
        "q": ball_record(q, config.serialization_digits),
        "V": ball_record(V, config.serialization_digits),
        "f_error_at_h0": ball_record(f_error, config.serialization_digits),
        "f_tube": ball_record(f_tube, config.serialization_digits),
        "y_tube": ball_record(y_tube, config.serialization_digits),
        "D_tube": ball_record(D_narrow, config.serialization_digits),
        "state_at_h0": {
            "f": ball_record(state_out[0], config.serialization_digits),
            "y": ball_record(state_out[1], config.serialization_digits),
        },
        "checks": {
            "q_below_one": True,
            "V_inside_trial_radius": True,
            "B_interval_contained": True,
            "G_y_negative": True,
            "D_positive": True,
        },
    }
    return state_out, D_narrow, record


def validate_standard_step(
    t0: fmpq,
    h: fmpq,
    state_in: tuple[arb, arb],
    config: ValidationConfig,
) -> tuple[tuple[arb, arb], arb, dict[str, Any]]:
    f_in, y_in = state_in
    f_center = rational_midpoint(f_in, config.state_center_digits)
    y_center = rational_midpoint(y_in, config.state_center_digits)
    f_R_cur = abs(f_in - arb(f_center)).abs_upper()
    y_R_cur = abs(y_in - arb(y_center)).abs_upper()
    R_cur = max_positive([f_R_cur, y_R_cur])

    rational_f, rational_y = standard_taylor_polynomial(
        t0,
        f_center,
        y_center,
        config.standard_order,
        config.coefficient_center_digits,
    )
    F = arb_poly([arb(coefficient) for coefficient in rational_f])
    Y = arb_poly([arb(coefficient) for coefficient in rational_y])
    T_polynomial = arb_poly([arb(t0), 1])
    S_polynomial = 1 - T_polynomial
    D_polynomial, N_polynomial = polynomial_dn(S_polynomial, F, Y)
    H_polynomial = N_polynomial - 9 * S_polynomial * Y * D_polynomial
    denominator_polynomial = 9 * T_polynomial * (2 - T_polynomial) * D_polynomial
    residual_numerator = Y.derivative() * denominator_polynomial - H_polynomial

    trial_radius = arb(decimal_to_fmpq(config.standard_trial_radius))
    L, D_tube, f_tube, y_tube = standard_path_bounds(
        F,
        Y,
        t0,
        h,
        trial_radius,
        config.standard_cells,
    )
    denominator_lower = 9 * arb(t0) * (2 - arb(t0 + h)) * D_tube.lower()
    if not denominator_lower > 0:
        raise ArithmeticError(
            f"standard residual denominator is nonpositive at t={t0}: "
            f"{denominator_lower}"
        )
    residual_numerator_bound = polynomial_l1_bound(residual_numerator, h)
    rho_step = (residual_numerator_bound / denominator_lower).abs_upper()

    R_next = ((R_cur + rho_step * arb(h)) * (L * arb(h)).exp()).abs_upper()
    if not trial_radius > R_next:
        raise ArithmeticError(
            f"step at t={t0} fails: R_next={R_next}, "
            f"trial radius={trial_radius}"
        )

    p_end_f = F(arb(h))
    p_end_y = Y(arb(h))
    state_out = (
        p_end_f + symmetric_ball(R_next),
        p_end_y + symmetric_ball(R_next),
    )

    record = {
        "t0": str(t0),
        "h": str(h),
        "t1": str(t0 + h),
        "center_in_f": str(f_center),
        "center_in_y": str(y_center),
        "order_y": config.standard_order,
        "order_f": config.standard_order + 1,
        "subdivision_cells": config.standard_cells,
        "trial_radius": config.standard_trial_radius,
        "R_cur": ball_record(R_cur, config.serialization_digits),
        "residual_numerator_bound": ball_record(
            residual_numerator_bound, config.serialization_digits
        ),
        "residual_denominator_lower": ball_record(
            denominator_lower, config.serialization_digits
        ),
        "rho_step": ball_record(rho_step, config.serialization_digits),
        "L": ball_record(L, config.serialization_digits),
        "R_next": ball_record(R_next, config.serialization_digits),
        "p_end": {
            "f": ball_record(p_end_f, config.serialization_digits),
            "y": ball_record(p_end_y, config.serialization_digits),
        },
        "f_tube": ball_record(f_tube, config.serialization_digits),
        "y_tube": ball_record(y_tube, config.serialization_digits),
        "D_tube": ball_record(D_tube, config.serialization_digits),
        "state_out": {
            "f": ball_record(state_out[0], config.serialization_digits),
            "y": ball_record(state_out[1], config.serialization_digits),
        },
        "checks": {
            "D_positive": True,
            "R_next_below_trial_radius": True,
        },
    }
    return state_out, D_tube, record


def _fmpq_float(value: fmpq) -> float:
    return float(Fraction(str(value)))


def run_validation(
    label: str,
    A_lower: fmpq,
    A_upper: fmpq,
    config: ValidationConfig = DEFAULT_CONFIG,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], tuple[arb, arb], arb]:
    config.apply()
    if A_upper < A_lower:
        raise ValueError("A_upper must not be below A_lower")

    state, initial_D, initial_record = validate_initial_step(
        A_lower, A_upper, config
    )
    D_global = initial_D
    standard_steps: list[dict[str, Any]] = []
    t = decimal_to_fmpq(config.initial_h)
    max_step = decimal_to_fmpq(config.max_standard_step)

    while t < 1:
        h = min(t / 4, max_step, 1 - t)
        state, D_step, step_record = validate_standard_step(t, h, state, config)
        D_global = D_global.union(D_step)
        step_number = len(standard_steps) + 1
        step_record["index"] = step_number
        standard_steps.append(step_record)
        t += h
        if progress and (step_number % 20 == 0 or t == 1):
            progress(
                f"{label}: step {step_number:03d}, "
                f"t={_fmpq_float(t):.12g}, "
                f"f={state[0].str(18)}, "
                f"R_next={step_record['R_next']['display']}"
            )

    run_record = {
        "label": label,
        "parameter_interval": {
            "A_lower": str(A_lower),
            "A_upper": str(A_upper),
        },
        "initial_step": initial_record,
        "standard_step_count": len(standard_steps),
        "standard_steps": standard_steps,
        "endpoint_t1_s0": {
            "f": ball_record(state[0], config.serialization_digits),
            "y": ball_record(state[1], config.serialization_digits),
        },
        "D_global_tube": ball_record(D_global, config.serialization_digits),
    }
    return run_record, state, D_global


# ---------------------------------------------------------------------------
# Compact default output and optional JSON output
# ---------------------------------------------------------------------------


def _fraction(text: str) -> Fraction:
    return Fraction(text)


def _floor(value: Fraction) -> int:
    return value.numerator // value.denominator


def _ceiling(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def _at_least_power_of_ten(value: Fraction, exponent: int) -> bool:
    numerator = abs(value.numerator)
    denominator = value.denominator
    if exponent >= 0:
        return numerator >= denominator * 10**exponent
    return numerator * 10 ** (-exponent) >= denominator


def _decimal_exponent(value: Fraction) -> int:
    if value == 0:
        raise ValueError("zero has no decimal exponent")
    exponent = len(str(abs(value.numerator))) - len(str(value.denominator))
    while not _at_least_power_of_ten(value, exponent):
        exponent -= 1
    while _at_least_power_of_ten(value, exponent + 1):
        exponent += 1
    return exponent


def _format_fraction(
    value: Fraction,
    *,
    significant_digits: int,
    round_up: bool,
) -> str:
    if value == 0:
        return "0"
    exponent = _decimal_exponent(value)
    unit_exponent = exponent - significant_digits + 1
    if unit_exponent >= 0:
        scaled = value / Fraction(10**unit_exponent, 1)
    else:
        scaled = value * Fraction(10 ** (-unit_exponent), 1)

    integer = _ceiling(scaled) if round_up else _floor(scaled)
    sign = "-" if integer < 0 else ""
    digits = str(abs(integer))
    scientific_exponent = unit_exponent + len(digits) - 1

    if -4 <= scientific_exponent < significant_digits:
        if unit_exponent >= 0:
            body = digits + "0" * unit_exponent
        else:
            decimal_position = len(digits) + unit_exponent
            if decimal_position > 0:
                body = digits[:decimal_position] + "." + digits[decimal_position:]
            else:
                body = "0." + "0" * (-decimal_position) + digits
        if "." in body:
            body = body.rstrip("0").rstrip(".")
        return sign + body

    mantissa = digits[0]
    if len(digits) > 1:
        mantissa += "." + digits[1:]
    return f"{sign}{mantissa}e{scientific_exponent:+d}"


def _record_lower(record: dict[str, Any]) -> Fraction:
    return _fraction(record["lower_rational"])


def _record_upper(record: dict[str, Any]) -> Fraction:
    return _fraction(record["upper_rational"])


def _format_interval(record: dict[str, Any], digits: int = 25) -> str:
    lower = _format_fraction(
        _record_lower(record), significant_digits=digits, round_up=False
    )
    upper = _format_fraction(
        _record_upper(record), significant_digits=digits, round_up=True
    )
    return f"[{lower}, {upper}]"


def _format_upper(record: dict[str, Any], digits: int = 25) -> str:
    return _format_fraction(
        _record_upper(record), significant_digits=digits, round_up=True
    )


def print_run_summary(run: dict[str, Any]) -> None:
    endpoint = run["endpoint_t1_s0"]
    initial = run["initial_step"]
    print(f"{run['label']}:")
    print(f"f(t=1) in {_format_interval(endpoint['f'])}")
    print(f"y(t=1) in {_format_interval(endpoint['y'])}")
    print(f"M_f <= {_format_upper(initial['M_f'])}")
    print(f"M_y <= {_format_upper(initial['M_y'])}")
    print(f"rho_0 <= {_format_upper(initial['rho_0'])}")
    print(f"q <= {_format_upper(initial['q'])}")
    print(f"V <= {_format_upper(initial['V'])}")


def _run_specs() -> list[tuple[str, str, str, str]]:
    return [
        ("A_minus", "A_-", A_MINUS_DECIMAL, A_MINUS_DECIMAL),
        ("A_plus", "A_+", A_PLUS_DECIMAL, A_PLUS_DECIMAL),
        ("A_interval", "[A_-, A_+]", A_MINUS_DECIMAL, A_PLUS_DECIMAL),
    ]


def _check_conditions(
    states: dict[str, tuple[arb, arb]],
    D_bounds: dict[str, arb],
) -> None:
    if not states["A_minus"][0] > 0:
        raise ArithmeticError("the A_- endpoint enclosure is not positive")
    if not states["A_plus"][0] < 0:
        raise ArithmeticError("the A_+ endpoint enclosure is not negative")
    for name, D_bound in D_bounds.items():
        if not D_bound > 0:
            raise ArithmeticError(f"{name}: D is not positive on the full tube")


def run_all(*, show_progress: bool, show_summaries: bool) -> dict[str, Any]:
    config = DEFAULT_CONFIG
    A_0 = decimal_to_fmpq(A_0_DECIMAL)
    runs: dict[str, Any] = {}
    states: dict[str, tuple[arb, arb]] = {}
    D_bounds: dict[str, arb] = {}

    progress = print if show_progress else None
    for key, label, lower_text, upper_text in _run_specs():
        run, state, D_bound = run_validation(
            label,
            decimal_to_fmpq(lower_text),
            decimal_to_fmpq(upper_text),
            config,
            progress,
        )
        runs[key] = run
        states[key] = state
        D_bounds[key] = D_bound
        if show_summaries:
            print_run_summary(run)

    _check_conditions(states, D_bounds)
    return {
        "schema": "special-lagrangian-shooting-steps-v1",
        "equation_coordinates": {
            "t": "1-s",
            "initial_interval": "0 <= t <= 1/100",
            "shooting_endpoint": "t=1 (equivalently s=0)",
        },
        "parameters": {
            "A_0_decimal": A_0_DECIMAL,
            "A_minus_decimal": A_MINUS_DECIMAL,
            "A_plus_decimal": A_PLUS_DECIMAL,
            "A_minus_equals_A_0_minus_1e_minus_26": (
                decimal_to_fmpq(A_MINUS_DECIMAL)
                == A_0 - decimal_to_fmpq("1e-26")
            ),
            "A_plus_equals_A_0_plus_1e_minus_26": (
                decimal_to_fmpq(A_PLUS_DECIMAL)
                == A_0 + decimal_to_fmpq("1e-26")
            ),
        },
        "configuration": config.as_dict(),
        "runs": runs,
        "conclusions": {
            "A_minus_f_at_t1": runs["A_minus"]["endpoint_t1_s0"]["f"],
            "A_plus_f_at_t1": runs["A_plus"]["endpoint_t1_s0"]["f"],
            "D_positive_on_all_tubes": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the three Cartan-cubic shooting calculations."
    )
    parser.add_argument(
        "--output-steps",
        action="store_true",
        help="print the full JSON step data and no progress summary",
    )
    args = parser.parse_args()

    if args.output_steps:
        data = run_all(show_progress=False, show_summaries=False)
        json.dump(data, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    run_all(show_progress=True, show_summaries=True)
    print("ALL THE CONDITIONS HAVE BEEN VALIDATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
