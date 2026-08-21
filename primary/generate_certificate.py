#!/usr/bin/env python3
"""Generate the rigorous JSON/CSV certificate for the shooting computation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import flint

from validator_core import (
    A_CENTER_DECIMAL,
    A_MINUS_DECIMAL,
    A_PLUS_DECIMAL,
    DEFAULT_CONFIG,
    D_DRAFT_THRESHOLD_DECIMAL,
    ball_record,
    decimal_to_fmpq,
    quick_certificate_checks,
    run_branch_validation,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_step_csv(path: Path, runs: dict[str, Any]) -> None:
    fields = [
        "run",
        "index",
        "x0",
        "h",
        "x1",
        "center_in_f",
        "center_in_y",
        "initial_error_upper",
        "residual_bound_rho_upper",
        "jacobian_bound_L_upper",
        "propagated_error_beta_upper",
        "trial_radius",
        "D_tube_lower",
        "f_tube_lower",
        "f_tube_upper",
        "y_tube_lower",
        "y_tube_upper",
        "state_out_f_lower",
        "state_out_f_upper",
        "state_out_y_lower",
        "state_out_y_upper",
        "polynomial_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run_name, run in runs.items():
            for step in run["continuation_steps"]:
                writer.writerow(
                    {
                        "run": run_name,
                        "index": step["index"],
                        "x0": step["x0"],
                        "h": step["h"],
                        "x1": step["x1"],
                        "center_in_f": step["center_in_f"],
                        "center_in_y": step["center_in_y"],
                        "initial_error_upper": step["initial_error"]["upper_rational"],
                        "residual_bound_rho_upper": step["residual_bound_rho"]["upper_rational"],
                        "jacobian_bound_L_upper": step["jacobian_bound_L"]["upper_rational"],
                        "propagated_error_beta_upper": step["propagated_error_beta"]["upper_rational"],
                        "trial_radius": step["trial_radius"],
                        "D_tube_lower": step["D_tube"]["lower_rational"],
                        "f_tube_lower": step["f_tube"]["lower_rational"],
                        "f_tube_upper": step["f_tube"]["upper_rational"],
                        "y_tube_lower": step["y_tube"]["lower_rational"],
                        "y_tube_upper": step["y_tube"]["upper_rational"],
                        "state_out_f_lower": step["state_out"]["f"]["lower_rational"],
                        "state_out_f_upper": step["state_out"]["f"]["upper_rational"],
                        "state_out_y_lower": step["state_out"]["y"]["lower_rational"],
                        "state_out_y_upper": step["state_out"]["y"]["upper_rational"],
                        "polynomial_sha256": step["polynomial_sha256"],
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory in which certificate files are written",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DEFAULT_CONFIG
    config.apply()
    A_minus = decimal_to_fmpq(A_MINUS_DECIMAL)
    A_plus = decimal_to_fmpq(A_PLUS_DECIMAL)

    def progress(message: str) -> None:
        print(message, flush=True)

    runs: dict[str, Any] = {}
    internal_results = {}
    for label, lower, upper in [
        ("A_minus", A_minus, A_minus),
        ("A_plus", A_plus, A_plus),
        ("A_interval", A_minus, A_plus),
    ]:
        print(f"starting {label}", flush=True)
        record, endpoint_state, D_global = run_branch_validation(
            label, lower, upper, config, progress
        )
        runs[label] = record
        internal_results[label] = (endpoint_state, D_global)

    minus_f = internal_results["A_minus"][0][0]
    plus_f = internal_results["A_plus"][0][0]
    uniform_D = internal_results["A_interval"][1]
    if not minus_f > 0:
        raise ArithmeticError(f"A_minus sign not certified: {minus_f}")
    if not plus_f < 0:
        raise ArithmeticError(f"A_plus sign not certified: {plus_f}")
    if not uniform_D > decimal_to_fmpq(D_DRAFT_THRESHOLD_DECIMAL):
        raise ArithmeticError(
            f"uniform D bound does not exceed {D_DRAFT_THRESHOLD_DECIMAL}: {uniform_D}"
        )

    generated_utc = datetime.now(timezone.utc).isoformat()
    certificate = {
        "schema": "special-lagrangian-cartancubic-certificate-v1",
        "generated_utc": generated_utc,
        "equation_coordinates": {
            "x": "1-s",
            "local_interval": "0 <= x <= 1/100",
            "endpoint_of_shooting": "x=1 (equivalently s=0)",
        },
        "exact_parameters": {
            "A_minus_decimal": A_MINUS_DECIMAL,
            "A_center_decimal": A_CENTER_DECIMAL,
            "A_plus_decimal": A_PLUS_DECIMAL,
        },
        "configuration": config.as_dict(),
        "method": {
            "interval_type": "Arb midpoint-radius balls with rigorous outward rounding",
            "local_validation": "singular integral-equation contraction around a rational Taylor polynomial",
            "continuation_validation": "rational Taylor residual plus an infinity-norm Gronwall bound",
            "D_validation": "subdivided natural interval evaluation on every accepted tube",
        },
        "runs": runs,
        "certified_claims": {
            "A_minus_f_at_s0": ball_record(minus_f, config.serialization_digits),
            "A_minus_sign": "strictly positive",
            "A_plus_f_at_s0": ball_record(plus_f, config.serialization_digits),
            "A_plus_sign": "strictly negative",
            "uniform_D_tube": ball_record(uniform_D, config.serialization_digits),
            "uniform_D_exceeds": D_DRAFT_THRESHOLD_DECIMAL,
        },
    }
    quick_messages = quick_certificate_checks(certificate)
    certificate["internal_exact_rational_checks"] = quick_messages

    certificate_path = output_dir / "certificate.json"
    endpoints_path = output_dir / "endpoint_enclosures.json"
    environment_path = output_dir / "environment.json"
    steps_path = output_dir / "continuation_steps.csv"
    write_json(certificate_path, certificate)
    write_step_csv(steps_path, runs)

    endpoint_payload = {
        "schema": certificate["schema"],
        "parameters": certificate["exact_parameters"],
        "A_minus": {
            "f_at_s0": certificate["certified_claims"]["A_minus_f_at_s0"],
            "certified_sign": "positive",
        },
        "A_plus": {
            "f_at_s0": certificate["certified_claims"]["A_plus_f_at_s0"],
            "certified_sign": "negative",
        },
        "uniform_parameter_interval": {
            "A_lower": A_MINUS_DECIMAL,
            "A_upper": A_PLUS_DECIMAL,
            "D_tube": certificate["certified_claims"]["uniform_D_tube"],
            "strictly_exceeds": D_DRAFT_THRESHOLD_DECIMAL,
        },
        "ball_encoding": "[mid_integer +/- radius_integer] * 10**exponent10; exact rational endpoints are also given",
    }
    write_json(endpoints_path, endpoint_payload)

    source_dir = Path(__file__).resolve().parent
    source_paths = [source_dir / "validator_core.py", source_dir / "generate_certificate.py"]
    environment = {
        "generated_utc": generated_utc,
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "libraries": {
            "python-flint": flint.__version__,
            "FLINT": getattr(flint, "__FLINT_VERSION__", "unknown"),
        },
        "working_precision_bits": config.precision_bits,
        "working_precision_decimal_digits_approx": int(config.precision_bits * 0.3010299956639812),
        "source_sha256": {path.name: file_sha256(path) for path in source_paths},
        "certificate_sha256": file_sha256(certificate_path),
        "continuation_steps_csv_sha256": file_sha256(steps_path),
        "endpoint_enclosures_sha256": file_sha256(endpoints_path),
    }
    write_json(environment_path, environment)

    (output_dir / "certificate.sha256").write_text(
        f"{file_sha256(certificate_path)}  certificate.json\n", encoding="ascii"
    )

    print("\nCERTIFIED", flush=True)
    print("A_minus f(s=0):", minus_f.str(70), flush=True)
    print("A_plus  f(s=0):", plus_f.str(70), flush=True)
    print("uniform D tube:", uniform_D.str(70), flush=True)
    print("certificate:", certificate_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
