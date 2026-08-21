#!/usr/bin/env python3
"""Second, independently written interval validation of the shooting proof.

The script shares no code with validator_core.py.  It reads exact rational
Taylor polynomials as witnesses, evaluates D and N from the complex K,T
formulas (rather than the expanded real polynomials), and recomputes all local
contraction, residual, Jacobian, Gronwall, endpoint-sign, and uniform-D
inequalities.  It uses a higher precision and twice as many path subdivisions
as the primary implementation.
"""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import flint
from flint import acb, acb_poly, arb, arb_poly, ctx, fmpq

PRECISION_BITS = 768
LOCAL_CELLS = 128
REGULAR_CELLS = 32
ctx.prec = PRECISION_BITS
ctx.threads = 1

def parse_q(value: Any) -> fmpq:
    text = str(value)
    try:
        return fmpq(text)
    except ValueError:
        fraction = Fraction(Decimal(text))
        return fmpq(fraction.numerator, fraction.denominator)


def hull(a: fmpq | str, b: fmpq | str) -> arb:
    return arb(parse_q(a)).union(arb(parse_q(b)))


def symmetric(radius: arb | fmpq | str) -> arb:
    r = radius if isinstance(radius, arb) else arb(parse_q(radius))
    return arb(0, r.abs_upper())


def abs_upper(value: arb) -> arb:
    return abs(value).abs_upper()


def max_upper(*values: arb) -> arb:
    out = arb(0)
    for value in values:
        out = out.union(abs_upper(value)).abs_upper()
    return out


def real_poly(poly: acb_poly) -> arb_poly:
    return arb_poly([coefficient.real for coefficient in poly.coeffs()])


def imag_poly(poly: acb_poly) -> arb_poly:
    return arb_poly([coefficient.imag for coefficient in poly.coeffs()])


def dn_derivatives(s: arb, f: arb, y: arb) -> tuple[arb, arb, arb, arb, arb, arb]:
    K = acb(arb(1), 2 * f)
    one_minus_s2 = 1 - s * s
    p2 = 9 * one_minus_s2 * y * y
    A = K**2 + p2
    T = K**3 + 3 * p2 * K - acb(0, 3) * s * y * (3 * K**2 + p2)
    D = (K * T).real
    N = (A * T).imag

    def derivative(dK: acb, dp2: arb, dy_var: arb) -> tuple[arb, arb]:
        dT = (
            3 * K**2 * dK
            + 3 * (dp2 * K + p2 * dK)
            - acb(0, 3)
            * s
            * (dy_var * (3 * K**2 + p2) + y * (6 * K * dK + dp2))
        )
        dD = (dK * T + K * dT).real
        dA = 2 * K * dK + dp2
        dN = (dA * T + A * dT).imag
        return dD, dN

    Df, Nf = derivative(acb(0, 2), arb(0), arb(0))
    Dy, Ny = derivative(acb(0), 18 * one_minus_s2 * y, arb(1))
    return D, N, Df, Dy, Nf, Ny


def gq_derivatives(x: arb, f: arb, y: arb, regular: bool) -> tuple[arb, arb, arb, arb]:
    s = 1 - x
    D, N, Df, Dy, Nf, Ny = dn_derivatives(s, f, y)
    H = N - 9 * s * y * D
    Hf = Nf - 9 * s * y * Df
    Hy = Ny - 9 * s * (D + y * Dy)
    factor = 9 * (2 - x)
    if regular:
        factor *= x
    denominator = factor * D
    denominator_f = factor * Df
    denominator_y = factor * Dy
    value = H / denominator
    df = (Hf * denominator - H * denominator_f) / denominator**2
    dy = (Hy * denominator - H * denominator_y) / denominator**2
    return value, df, dy, D


def polynomial_dn_complex(S: arb_poly, F: arb_poly, Y: arb_poly) -> tuple[arb_poly, arb_poly]:
    K = acb_poly([1]) + acb(0, 2) * acb_poly(F)
    p2 = 9 * (1 - S**2) * Y**2
    p2c = acb_poly(p2)
    T = K**3 + 3 * p2c * K - acb(0, 3) * acb_poly(S * Y) * (3 * K**2 + p2c)
    return real_poly(K * T), imag_poly((K**2 + p2c) * T)


def residual_polynomial(F: arb_poly, Y: arb_poly, x0: fmpq, regular: bool) -> arb_poly:
    if regular:
        X = arb_poly([arb(x0), 1])
        S = 1 - X
    else:
        X = arb_poly([0, 1])
        S = arb_poly([1, -1])
    D, N = polynomial_dn_complex(S, F, Y)
    H = N - 9 * S * Y * D
    if regular:
        denominator = 9 * X * (2 - X) * D
        return Y.derivative() * denominator - H
    denominator = 9 * (2 - X) * D
    return X * Y.derivative() * denominator - H


def polynomial_l1_bound(poly: arb_poly, h: fmpq) -> arb:
    total = arb(0)
    power = arb(1)
    hb = arb(h)
    for coefficient in poly.coeffs():
        total += abs(coefficient) * power
        power *= hb
    return total.abs_upper()


def polynomial_hash(f_strings: Sequence[str], y_strings: Sequence[str]) -> str:
    payload = "F\n" + "\n".join(f_strings) + "\nY\n" + "\n".join(y_strings)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def record_interval(record: dict[str, Any]) -> arb:
    return hull(record["lower_rational"], record["upper_rational"])


def ball_json(value: arb, digits: int = 100) -> dict[str, str]:
    lo = value.lower().str(digits, radius=False)
    hi = value.upper().str(digits, radius=False)
    return {"lower_decimal": lo, "upper_decimal": hi, "display": value.str(60)}


def local_validate(run_name: str, run: dict[str, Any], witness: dict[str, Any], config: dict[str, Any]) -> tuple[tuple[arb, arb], arb, dict[str, Any]]:
    local = run["local_taylor_validation"]
    fs = witness["local"]["f_coefficients_rational"]
    ys = witness["local"]["y_coefficients_rational"]
    if polynomial_hash(fs, ys) != local["polynomial_sha256"]:
        raise AssertionError(f"{run_name}: local polynomial hash mismatch")
    F = arb_poly([arb(parse_q(value)) for value in fs])
    Y = arb_poly([arb(parse_q(value)) for value in ys])
    h = parse_q(config["local_h"])
    trial = arb(parse_q(config["local_trial_radius"]))
    A_radius = arb(parse_q(local["A_radius"]))

    A_box = hull(local["A_lower"], local["A_upper"])
    B_box = record_interval(local["B_interval"])
    D_box, N_box, _Df, Dy_box, _Nf, Ny_box = dn_derivatives(arb(1), A_box, B_box)
    Hy_box = Ny_box - 9 * (D_box + B_box * Dy_box)
    if not Hy_box < 0:
        raise AssertionError(f"{run_name}: endpoint H_y is not negative: {Hy_box}")
    B_formula = (2 * A_box + ((arb(3) / 2) * (2 * A_box).atan()).tan()) / 9
    if not B_box.contains(B_formula):
        raise AssertionError(
            f"{run_name}: explicit endpoint slope is not contained in the B box"
        )

    residual_num = polynomial_l1_bound(residual_polynomial(F, Y, fmpq(0), False), h)
    coarse_f_error = (A_radius + arb(h) * trial).abs_upper()
    Mf = arb(0)
    My = arb(0)
    D_coarse: arb | None = None
    for cell in range(LOCAL_CELLS):
        t = hull(h * cell / LOCAL_CELLS, h * (cell + 1) / LOCAL_CELLS)
        fbox = F(t) + symmetric(coarse_f_error)
        ybox = Y(t) + symmetric(trial)
        _value, gf, gy, Dbox = gq_derivatives(t, fbox, ybox, False)
        if not Dbox > 0:
            raise AssertionError(f"{run_name}: local coarse D failed, cell {cell}")
        Mf = Mf.union(abs_upper(gf)).abs_upper()
        My = My.union(abs_upper(gy + 1)).abs_upper()
        D_coarse = Dbox if D_coarse is None else D_coarse.union(Dbox)
    assert D_coarse is not None
    denominator_lower = 9 * (2 - arb(h)) * D_coarse.lower()
    Rs = (residual_num / denominator_lower).abs_upper()
    q = (arb(h) * Mf + My).abs_upper()
    if not q < 1:
        raise AssertionError(f"{run_name}: local q failed: {q}")
    V = ((Mf * A_radius + Rs) / (1 - q)).abs_upper()
    if not V < trial:
        raise AssertionError(f"{run_name}: local V failed: {V}")
    f_error = (A_radius + arb(h) * V).abs_upper()

    D_narrow: arb | None = None
    for cell in range(LOCAL_CELLS):
        t = hull(h * cell / LOCAL_CELLS, h * (cell + 1) / LOCAL_CELLS)
        fbox = F(t) + symmetric(f_error)
        ybox = Y(t) + symmetric(V)
        Dbox, *_ = dn_derivatives(1 - t, fbox, ybox)
        if not Dbox > 0:
            raise AssertionError(f"{run_name}: local narrow D failed, cell {cell}")
        D_narrow = Dbox if D_narrow is None else D_narrow.union(Dbox)
    assert D_narrow is not None
    state = (F(arb(h)) + symmetric(f_error), Y(arb(h)) + symmetric(V))
    return state, D_narrow, {
        "endpoint_Hy": ball_json(Hy_box),
        "endpoint_B_formula": ball_json(B_formula),
        "residual_numerator": ball_json(residual_num),
        "residual_denominator_lower": ball_json(denominator_lower),
        "M_f": ball_json(Mf),
        "M_y_plus_one": ball_json(My),
        "contraction_q": ball_json(q),
        "tail_y": ball_json(V),
        "f_error_at_h": ball_json(f_error),
        "D_tube": ball_json(D_narrow),
    }


def regular_validate(run_name: str, run: dict[str, Any], witness: dict[str, Any], state: tuple[arb, arb], D_global: arb, config: dict[str, Any]) -> tuple[tuple[arb, arb], arb, list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    for step, poly in zip(run["continuation_steps"], witness["steps"], strict=True):
        index = int(step["index"])
        fs = poly["f_coefficients_rational"]
        ys = poly["y_coefficients_rational"]
        if poly["index"] != index or polynomial_hash(fs, ys) != step["polynomial_sha256"]:
            raise AssertionError(f"{run_name} step {index}: witness mismatch")
        F = arb_poly([arb(parse_q(value)) for value in fs])
        Y = arb_poly([arb(parse_q(value)) for value in ys])
        x0 = parse_q(step["x0"])
        h = parse_q(step["h"])
        radius = arb(parse_q(step["trial_radius"]))
        fc = parse_q(step["center_in_f"])
        yc = parse_q(step["center_in_y"])
        E0 = max_upper(state[0] - arb(fc), state[1] - arb(yc))

        residual_num = polynomial_l1_bound(residual_polynomial(F, Y, x0, True), h)
        L = arb(0)
        D_step: arb | None = None
        for cell in range(REGULAR_CELLS):
            t = hull(h * cell / REGULAR_CELLS, h * (cell + 1) / REGULAR_CELLS)
            xbox = arb(x0) + t
            fbox = F(t) + symmetric(radius)
            ybox = Y(t) + symmetric(radius)
            _value, qf, qy, Dbox = gq_derivatives(xbox, fbox, ybox, True)
            if not Dbox > 0:
                raise AssertionError(f"{run_name} step {index}: D failed, cell {cell}")
            L = L.union((1 + abs(qf) + abs(qy)).abs_upper()).abs_upper()
            D_step = Dbox if D_step is None else D_step.union(Dbox)
        assert D_step is not None
        D_global = D_global.union(D_step)
        denominator_lower = 9 * arb(x0) * (2 - arb(x0 + h)) * D_step.lower()
        rho = (residual_num / denominator_lower).abs_upper()
        beta = ((E0 + rho * arb(h)) * (L * arb(h)).exp()).abs_upper()
        if not beta < radius:
            raise AssertionError(f"{run_name} step {index}: beta failed: {beta}")
        state = (F(arb(h)) + symmetric(beta), Y(arb(h)) + symmetric(beta))
        output.append(
            {
                "index": index,
                "x0": str(x0),
                "h": str(h),
                "initial_error": ball_json(E0),
                "residual_numerator": ball_json(residual_num),
                "residual_denominator_lower": ball_json(denominator_lower),
                "rho": ball_json(rho),
                "L": ball_json(L),
                "beta": ball_json(beta),
                "D_tube": ball_json(D_step),
            }
        )
        if index % 20 == 0 or index == len(run["continuation_steps"]) - 1:
            print(
                f"{run_name}: step {index:03d}, beta={beta.str(12)}, D={D_step.str(12)}",
                flush=True,
            )
    return state, D_global, output


def main() -> int:
    here = Path(__file__).resolve().parent
    certificate = json.loads((here / "certificate.json").read_text(encoding="utf-8"))
    witness = json.loads((here / "witness_polynomials.json").read_text(encoding="utf-8"))
    config = certificate["configuration"]
    result: dict[str, Any] = {
        "schema": "special-lagrangian-cartancubic-independent-arb-validation-v1",
        "python_flint_version": flint.__version__,
        "flint_version": getattr(flint, "__FLINT_VERSION__", "unknown"),
        "precision_bits": PRECISION_BITS,
        "local_cells": LOCAL_CELLS,
        "regular_cells": REGULAR_CELLS,
        "D_N_formula": "complex K,T representation",
        "imports_primary_validator": False,
        "runs": {},
    }
    for run_name in ["A_minus", "A_plus", "A_interval"]:
        print(f"validating {run_name}", flush=True)
        run = certificate["runs"][run_name]
        run_witness = witness["runs"][run_name]
        state, D_global, local_result = local_validate(run_name, run, run_witness, config)
        state, D_global, steps = regular_validate(run_name, run, run_witness, state, D_global, config)
        if run_name == "A_minus" and not state[0] > 0:
            raise AssertionError(f"A_minus endpoint sign failed: {state[0]}")
        if run_name == "A_plus" and not state[0] < 0:
            raise AssertionError(f"A_plus endpoint sign failed: {state[0]}")
        if run_name == "A_interval" and not D_global.lower() > arb("4.2964512605"):
            raise AssertionError(f"uniform D bound failed: {D_global}")
        result["runs"][run_name] = {
            "local": local_result,
            "steps": steps,
            "endpoint_f": ball_json(state[0]),
            "endpoint_y": ball_json(state[1]),
            "D_global": ball_json(D_global),
        }
        print(
            f"PASS {run_name}: f(0)={state[0].str(35)}, D_global={D_global.str(35)}",
            flush=True,
        )
    destination = here / "independent_arb_results.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("INDEPENDENT ARB IMPLEMENTATION PASSED", flush=True)
    print(f"wrote {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
