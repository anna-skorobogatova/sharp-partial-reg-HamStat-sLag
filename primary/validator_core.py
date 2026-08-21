#!/usr/bin/env python3
"""Rigorous ball-arithmetic validator for the Cartan-cubic shooting problem.

The proof mechanism has two parts.

1. On x in [0, h0], where x = 1-s, a rational Taylor polynomial p is
   validated by a contraction argument for the singular Briot--Bouquet
   integral equation.  The reported numbers M_f, M_y, q and R_s give the
   explicit tail bound

       V = (M_f U_0 + R_s) / (1-q),   q = h0 M_f + M_y < 1,

   with |y-p_y| <= V and |f-p_f| <= U_0 + h0 V.

2. On every regular continuation step, a rational Taylor polynomial p is
   validated by an a-posteriori residual estimate and Gronwall's inequality.
   If E_0 bounds the incoming infinity-norm error, rho bounds |p'-F(x,p)|,
   and L bounds the infinity norm of the Jacobian of F on a trial tube, then

       beta = exp(L h) (E_0 + rho h)

   bounds the error on the whole step.  The step is accepted only if
   beta is strictly smaller than the trial tube radius and D is strictly
   positive throughout that tube.

All numerical inequalities are evaluated with python-flint/Arb ball
arithmetic.  Exact decimal input data and polynomial centers are stored as
rational numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
from typing import Any, Callable, Iterable, Sequence

from flint import arb, arb_poly, ctx, fmpq


A_MINUS_DECIMAL = "0.66922990609204402834133929821078023454"
A_CENTER_DECIMAL = "0.66922990609204402834133930821078023454"
A_PLUS_DECIMAL = "0.66922990609204402834133931821078023454"
D_DRAFT_THRESHOLD_DECIMAL = "4.2964512605"


@dataclass(frozen=True)
class ValidationConfig:
    precision_bits: int = 512
    local_h: str = "0.01"
    local_order: int = 40
    regular_order: int = 30
    local_cells: int = 64
    regular_cells: int = 16
    local_trial_radius: str = "1e-12"
    regular_trial_radius: str = "1e-10"
    max_regular_step: str = "0.01"
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
            "local_h": self.local_h,
            "local_order": self.local_order,
            "regular_order": self.regular_order,
            "local_cells": self.local_cells,
            "regular_cells": self.regular_cells,
            "local_trial_radius": self.local_trial_radius,
            "regular_trial_radius": self.regular_trial_radius,
            "max_regular_step": self.max_regular_step,
            "regular_step_rule": "h=min(x/4,max_regular_step,1-x)",
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
    midpoint digits even when the input ball itself has fewer certified digits.
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


def polynomial_hash(f_coefficients: Sequence[fmpq], y_coefficients: Sequence[fmpq]) -> str:
    payload = (
        "F\n"
        + "\n".join(str(item) for item in f_coefficients)
        + "\nY\n"
        + "\n".join(str(item) for item in y_coefficients)
    )
    return sha256(payload.encode("ascii")).hexdigest()


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


def singular_rhs_G(x: Any, f: Any, y: Any) -> Any:
    s = 1 - x
    D, N = dn_value(s, f, y)
    return (N - 9 * s * y * D) / (9 * (2 - x) * D)


def regular_rhs_Q(x: Any, f: Any, y: Any) -> Any:
    s = 1 - x
    D, N = dn_value(s, f, y)
    return (N - 9 * s * y * D) / (9 * x * (2 - x) * D)


def endpoint_slope_B(A: arb) -> arb:
    return (2 * A + ((arb(3) / 2) * (2 * A).atan()).tan()) / 9


# ---------------------------------------------------------------------------
# Rational polynomial construction
# ---------------------------------------------------------------------------


def singular_taylor_polynomial(
    A_center: fmpq,
    y0_center: fmpq,
    order: int,
    coefficient_digits: int,
) -> tuple[list[fmpq], list[fmpq], arb]:
    """Construct a rationalized local Taylor polynomial.

    The temporary Arb coefficients solve the formal recurrence.  They are then
    rounded to exact decimal rationals.  The final proof does not trust the
    recurrence: it bounds the residual of the resulting rational polynomial.
    """
    f_coefficients = [arb(0)] * (order + 2)
    y_coefficients = [arb(0)] * (order + 1)
    f_coefficients[0] = arb(A_center)
    y_coefficients[0] = arb(y0_center)

    endpoint_dual = singular_rhs_G(
        arb(0),
        Dual(arb(A_center), 1, 0),
        Dual(arb(y0_center), 0, 1),
    )
    base_gy = endpoint_dual.dy

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
        y_coefficients[k] = G[k] / (k - base_gy)

    rational_y = [
        rational_midpoint(coefficient, coefficient_digits)
        for coefficient in y_coefficients
    ]
    rational_y[0] = y0_center
    rational_f = [A_center] + [
        -rational_y[index] / (index + 1) for index in range(order + 1)
    ]
    return rational_f, rational_y, base_gy


def regular_taylor_polynomial(
    x0: fmpq,
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
        x_series = [arb(x0), arb(1)] + [arb(0)] * max(0, k - 1)
        s_series = [arb(1 - x0), arb(-1)] + [arb(0)] * max(0, k - 1)
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
                    x_series,
                    [arb(2 - x0), arb(-1)],
                    k,
                ),
                D,
                k,
            ),
            9,
            k,
        )
        Q = series_divide(H, denominator, k)
        y_coefficients[k + 1] = Q[k] / (k + 1)

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


def local_path_bounds(
    F: arb_poly,
    Y: arb_poly,
    h: fmpq,
    f_error: arb,
    y_error: arb,
    cells: int,
) -> tuple[arb, arb, arb, arb, arb]:
    M_f = arb(0)
    M_y_plus_one = arb(0)
    D_all: arb | None = None
    f_all: arb | None = None
    y_all: arb | None = None

    for cell in range(cells):
        left = h * cell / cells
        right = h * (cell + 1) / cells
        t_interval = interval_hull(left, right)
        x_interval = t_interval
        s_interval = 1 - x_interval
        f_box = F(t_interval) + symmetric_ball(f_error)
        y_box = Y(t_interval) + symmetric_ball(y_error)
        D_cell, _ = dn_value(s_interval, f_box, y_box)
        if not D_cell > 0:
            raise ArithmeticError(
                f"local D interval contains a nonpositive value in cell {cell}: {D_cell}"
            )

        derivative = singular_rhs_G(
            x_interval,
            Dual(f_box, 1, 0),
            Dual(y_box, 0, 1),
        )
        M_f = M_f.union(abs(derivative.df).abs_upper()).abs_upper()
        M_y_plus_one = M_y_plus_one.union(
            abs(derivative.dy + 1).abs_upper()
        ).abs_upper()
        D_all = D_cell if D_all is None else D_all.union(D_cell)
        f_all = f_box if f_all is None else f_all.union(f_box)
        y_all = y_box if y_all is None else y_all.union(y_box)

    assert D_all is not None and f_all is not None and y_all is not None
    return M_f, M_y_plus_one, D_all, f_all, y_all


def regular_path_bounds(
    F: arb_poly,
    Y: arb_poly,
    x0: fmpq,
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
        t_interval = interval_hull(left, right)
        x_interval = arb(x0) + t_interval
        s_interval = 1 - x_interval
        f_box = F(t_interval) + symmetric_ball(radius)
        y_box = Y(t_interval) + symmetric_ball(radius)
        D_cell, _ = dn_value(s_interval, f_box, y_box)
        if not D_cell > 0:
            raise ArithmeticError(
                f"regular D interval contains a nonpositive value at x={x0}, "
                f"cell={cell}: {D_cell}"
            )

        derivative = regular_rhs_Q(
            x_interval,
            Dual(f_box, 1, 0),
            Dual(y_box, 0, 1),
        )
        # Infinity-norm Jacobian bound.  max(1, |Q_f|+|Q_y|) is bounded by
        # the slightly simpler 1+|Q_f|+|Q_y|.
        L_cell = (1 + abs(derivative.df) + abs(derivative.dy)).abs_upper()
        L_all = L_all.union(L_cell).abs_upper()
        D_all = D_cell if D_all is None else D_all.union(D_cell)
        f_all = f_box if f_all is None else f_all.union(f_box)
        y_all = y_box if y_all is None else y_all.union(y_box)

    assert D_all is not None and f_all is not None and y_all is not None
    return L_all, D_all, f_all, y_all


# ---------------------------------------------------------------------------
# Local singular validation and regular continuation
# ---------------------------------------------------------------------------


def validate_local_segment(
    A_lower: fmpq,
    A_upper: fmpq,
    config: ValidationConfig,
) -> tuple[tuple[arb, arb], arb, dict[str, Any]]:
    h = decimal_to_fmpq(config.local_h)
    A_center = (A_lower + A_upper) / 2
    A_radius = (A_upper - A_lower) / 2
    A_interval = interval_hull(A_lower, A_upper)
    B_interval = endpoint_slope_B(A_interval)
    B_center_ball = endpoint_slope_B(arb(A_center))
    y0_center = rational_midpoint(
        B_center_ball, config.endpoint_slope_center_digits
    )

    rational_f, rational_y, base_gy = singular_taylor_polynomial(
        A_center,
        y0_center,
        config.local_order,
        config.coefficient_center_digits,
    )
    F = arb_poly([arb(coefficient) for coefficient in rational_f])
    Y = arb_poly([arb(coefficient) for coefficient in rational_y])

    X_polynomial = arb_poly([0, 1])
    S_polynomial = arb_poly([1, -1])
    D_polynomial, N_polynomial = polynomial_dn(S_polynomial, F, Y)
    H_polynomial = N_polynomial - 9 * S_polynomial * Y * D_polynomial
    denominator_polynomial = 9 * arb_poly([2, -1]) * D_polynomial
    residual_numerator = (
        X_polynomial * Y.derivative() * denominator_polynomial - H_polynomial
    )

    trial_radius = arb(decimal_to_fmpq(config.local_trial_radius))
    coarse_f_error = (arb(A_radius) + arb(h) * trial_radius).abs_upper()
    M_f, M_y_plus_one, D_coarse, _coarse_f, _coarse_y = local_path_bounds(
        F,
        Y,
        h,
        coarse_f_error,
        trial_radius,
        config.local_cells,
    )
    contraction_q = (arb(h) * M_f + M_y_plus_one).abs_upper()
    if not contraction_q < 1:
        raise ArithmeticError(f"local contraction constant is not below one: {contraction_q}")

    denominator_lower = 9 * (2 - arb(h)) * D_coarse.lower()
    if not denominator_lower > 0:
        raise ArithmeticError(f"local residual denominator is not positive: {denominator_lower}")
    residual_numerator_bound = polynomial_l1_bound(residual_numerator, h)
    residual_bound = (residual_numerator_bound / denominator_lower).abs_upper()

    y_tail = (
        (M_f * arb(A_radius) + residual_bound) / (1 - contraction_q)
    ).abs_upper()
    if not trial_radius > y_tail:
        raise ArithmeticError(
            f"local tail {y_tail} does not fit in trial radius {trial_radius}"
        )
    f_error = (arb(A_radius) + arb(h) * y_tail).abs_upper()

    y0_box = arb(y0_center) + symmetric_ball(y_tail)
    if not y0_box.contains(B_interval):
        raise ArithmeticError(
            "the validated endpoint y-box does not contain the explicit B(A) interval"
        )
    # Since |G_y+1| < 1, G_y is strictly negative on the local tube; hence
    # the root in the y-box is unique and is the B(A) branch.
    if not M_y_plus_one < 1:
        raise ArithmeticError("G_y is not certified negative on the local tube")

    _Mf_narrow, _My_narrow, D_narrow, f_tube, y_tube = local_path_bounds(
        F,
        Y,
        h,
        f_error,
        y_tail,
        config.local_cells,
    )

    state_out = (
        F(arb(h)) + symmetric_ball(f_error),
        Y(arb(h)) + symmetric_ball(y_tail),
    )

    record = {
        "x_interval": ["0", str(h)],
        "A_lower": str(A_lower),
        "A_upper": str(A_upper),
        "A_center": str(A_center),
        "A_radius": str(A_radius),
        "B_interval": ball_record(B_interval, config.serialization_digits),
        "B_polynomial_center": str(y0_center),
        "order_y": config.local_order,
        "order_f": config.local_order + 1,
        "subdivision_cells": config.local_cells,
        "trial_radius": config.local_trial_radius,
        "base_G_y": ball_record(base_gy, config.serialization_digits),
        "f_coefficients_rational": [str(item) for item in rational_f],
        "y_coefficients_rational": [str(item) for item in rational_y],
        "polynomial_sha256": polynomial_hash(rational_f, rational_y),
        "residual_numerator_bound": ball_record(
            residual_numerator_bound, config.serialization_digits
        ),
        "residual_denominator_lower": ball_record(
            denominator_lower, config.serialization_digits
        ),
        "residual_bound_Rs": ball_record(residual_bound, config.serialization_digits),
        "M_f": ball_record(M_f, config.serialization_digits),
        "M_y_plus_one": ball_record(M_y_plus_one, config.serialization_digits),
        "contraction_q": ball_record(contraction_q, config.serialization_digits),
        "tail_y_V": ball_record(y_tail, config.serialization_digits),
        "total_f_error_at_h": ball_record(f_error, config.serialization_digits),
        "f_tube": ball_record(f_tube, config.serialization_digits),
        "y_tube": ball_record(y_tube, config.serialization_digits),
        "D_tube": ball_record(D_narrow, config.serialization_digits),
        "state_at_h": {
            "f": ball_record(state_out[0], config.serialization_digits),
            "y": ball_record(state_out[1], config.serialization_digits),
        },
        "checks": {
            "q_strictly_below_one": True,
            "tail_fits_trial_tube": True,
            "B_interval_contained": True,
            "G_y_strictly_negative": True,
            "D_strictly_positive": True,
        },
    }
    return state_out, D_narrow, record


def validate_regular_step(
    x0: fmpq,
    h: fmpq,
    state_in: tuple[arb, arb],
    config: ValidationConfig,
) -> tuple[tuple[arb, arb], arb, dict[str, Any]]:
    f_in, y_in = state_in
    f_center = rational_midpoint(f_in, config.state_center_digits)
    y_center = rational_midpoint(y_in, config.state_center_digits)
    f_initial_error = abs(f_in - arb(f_center)).abs_upper()
    y_initial_error = abs(y_in - arb(y_center)).abs_upper()
    initial_error = max_positive([f_initial_error, y_initial_error])

    rational_f, rational_y = regular_taylor_polynomial(
        x0,
        f_center,
        y_center,
        config.regular_order,
        config.coefficient_center_digits,
    )
    F = arb_poly([arb(coefficient) for coefficient in rational_f])
    Y = arb_poly([arb(coefficient) for coefficient in rational_y])
    X_polynomial = arb_poly([arb(x0), 1])
    S_polynomial = 1 - X_polynomial
    D_polynomial, N_polynomial = polynomial_dn(S_polynomial, F, Y)
    H_polynomial = N_polynomial - 9 * S_polynomial * Y * D_polynomial
    denominator_polynomial = 9 * X_polynomial * (2 - X_polynomial) * D_polynomial
    residual_numerator = Y.derivative() * denominator_polynomial - H_polynomial

    trial_radius = arb(decimal_to_fmpq(config.regular_trial_radius))
    L, D_tube, f_tube, y_tube = regular_path_bounds(
        F,
        Y,
        x0,
        h,
        trial_radius,
        config.regular_cells,
    )
    denominator_lower = 9 * arb(x0) * (2 - arb(x0 + h)) * D_tube.lower()
    if not denominator_lower > 0:
        raise ArithmeticError(
            f"regular residual denominator is nonpositive at x={x0}: {denominator_lower}"
        )
    residual_numerator_bound = polynomial_l1_bound(residual_numerator, h)
    residual_bound = (residual_numerator_bound / denominator_lower).abs_upper()

    propagated_error = (
        (initial_error + residual_bound * arb(h)) * (L * arb(h)).exp()
    ).abs_upper()
    if not trial_radius > propagated_error:
        raise ArithmeticError(
            f"step at x={x0} fails: beta={propagated_error}, "
            f"trial radius={trial_radius}"
        )

    p_end_f = F(arb(h))
    p_end_y = Y(arb(h))
    state_out = (
        p_end_f + symmetric_ball(propagated_error),
        p_end_y + symmetric_ball(propagated_error),
    )

    record = {
        "x0": str(x0),
        "h": str(h),
        "x1": str(x0 + h),
        "center_in_f": str(f_center),
        "center_in_y": str(y_center),
        "order_y": config.regular_order,
        "order_f": config.regular_order + 1,
        "subdivision_cells": config.regular_cells,
        "trial_radius": config.regular_trial_radius,
        "polynomial_sha256": polynomial_hash(rational_f, rational_y),
        "initial_error": ball_record(initial_error, config.serialization_digits),
        "residual_numerator_bound": ball_record(
            residual_numerator_bound, config.serialization_digits
        ),
        "residual_denominator_lower": ball_record(
            denominator_lower, config.serialization_digits
        ),
        "residual_bound_rho": ball_record(residual_bound, config.serialization_digits),
        "jacobian_bound_L": ball_record(L, config.serialization_digits),
        "propagated_error_beta": ball_record(
            propagated_error, config.serialization_digits
        ),
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
            "D_strictly_positive": True,
            "beta_strictly_below_trial_radius": True,
        },
    }
    return state_out, D_tube, record


def run_branch_validation(
    label: str,
    A_lower: fmpq,
    A_upper: fmpq,
    config: ValidationConfig = DEFAULT_CONFIG,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], tuple[arb, arb], arb]:
    config.apply()
    if A_upper < A_lower:
        raise ValueError("A_upper must not be below A_lower")

    state, local_D, local_record = validate_local_segment(A_lower, A_upper, config)
    D_global = local_D
    continuation: list[dict[str, Any]] = []
    x = decimal_to_fmpq(config.local_h)
    max_step = decimal_to_fmpq(config.max_regular_step)
    step_index = 0

    while x < 1:
        h = min(x / 4, max_step, 1 - x)
        state, D_step, step_record = validate_regular_step(x, h, state, config)
        D_global = D_global.union(D_step)
        step_record["index"] = step_index
        continuation.append(step_record)
        x += h
        if progress and (step_index % 20 == 0 or x == 1):
            progress(
                f"{label}: step {step_index:03d}, x={x}, "
                f"f={state[0].str(18)}, beta={step_record['propagated_error_beta']['display']}"
            )
        step_index += 1

    branch_record = {
        "label": label,
        "parameter_interval": {
            "A_lower": str(A_lower),
            "A_upper": str(A_upper),
        },
        "local_taylor_validation": local_record,
        "continuation_step_count": len(continuation),
        "continuation_steps": continuation,
        "endpoint_x1_s0": {
            "f": ball_record(state[0], config.serialization_digits),
            "y": ball_record(state[1], config.serialization_digits),
        },
        "D_global_tube": ball_record(D_global, config.serialization_digits),
    }
    return branch_record, state, D_global


def quick_certificate_checks(certificate: dict[str, Any]) -> list[str]:
    """Check exact-rational consequences already recorded in a certificate."""
    messages: list[str] = []
    runs = certificate["runs"]
    minus_f = runs["A_minus"]["endpoint_x1_s0"]["f"]
    plus_f = runs["A_plus"]["endpoint_x1_s0"]["f"]
    minus_lower = fmpq(minus_f["lower_rational"])
    plus_upper = fmpq(plus_f["upper_rational"])
    if not minus_lower > 0:
        raise ArithmeticError("A_minus endpoint lower bound is not positive")
    if not plus_upper < 0:
        raise ArithmeticError("A_plus endpoint upper bound is not negative")
    messages.append("endpoint signs are strict")

    D_record = runs["A_interval"]["D_global_tube"]
    D_lower = fmpq(D_record["lower_rational"])
    threshold = decimal_to_fmpq(D_DRAFT_THRESHOLD_DECIMAL)
    if not D_lower > threshold:
        raise ArithmeticError(
            f"uniform D lower bound {D_lower} does not exceed {threshold}"
        )
    messages.append("uniform D lower bound exceeds the draft threshold")

    for run_name, run in runs.items():
        local = run["local_taylor_validation"]
        q_upper = fmpq(local["contraction_q"]["upper_rational"])
        if not q_upper < 1:
            raise ArithmeticError(f"{run_name}: local q is not below one")
        for step in run["continuation_steps"]:
            beta_upper = fmpq(step["propagated_error_beta"]["upper_rational"])
            trial = decimal_to_fmpq(step["trial_radius"])
            if not beta_upper < trial:
                raise ArithmeticError(
                    f"{run_name}, step {step['index']}: beta is not below trial radius"
                )
    messages.append("all local contractions and continuation tube inequalities are strict")
    return messages
