#!/usr/bin/env python3
"""Verify the archived Cartan-cubic special-Lagrangian certificate.

Two verification levels are available.

* The default quick verification uses exact rational arithmetic to check the
  serialization, hashes, step chain, strict endpoint signs, local contraction
  inequalities, continuation-tube inequalities, and the global D lower bound.
* ``--recompute`` reruns every Arb validation step from the source equations
  and checks that the recomputed enclosures are contained in the archived
  machine-readable enclosures and that every polynomial hash agrees.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import flint
from flint import fmpq

from validator_core import (
    A_MINUS_DECIMAL,
    A_PLUS_DECIMAL,
    DEFAULT_CONFIG,
    D_DRAFT_THRESHOLD_DECIMAL,
    ball_from_record,
    decimal_to_fmpq,
    quick_certificate_checks,
    rational_from_scaled_integer,
    run_branch_validation,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def is_ball_record(value: Any) -> bool:
    return isinstance(value, dict) and {
        "mid_integer",
        "radius_integer",
        "exponent10",
        "lower_rational",
        "upper_rational",
    }.issubset(value)


def walk_ball_records(value: Any, path: str = "certificate") -> Iterable[tuple[str, dict[str, Any]]]:
    if is_ball_record(value):
        yield path, value
        return
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk_ball_records(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk_ball_records(child, f"{path}[{index}]")


def verify_ball_serialization(certificate: dict[str, Any]) -> int:
    count = 0
    for path, record in walk_ball_records(certificate):
        midpoint = int(record["mid_integer"])
        radius = int(record["radius_integer"])
        exponent = int(record["exponent10"])
        require(radius >= 0, f"{path}: negative serialized radius")
        expected_lower = rational_from_scaled_integer(midpoint - radius, exponent)
        expected_upper = rational_from_scaled_integer(midpoint + radius, exponent)
        actual_lower = fmpq(record["lower_rational"])
        actual_upper = fmpq(record["upper_rational"])
        require(actual_lower == expected_lower, f"{path}: inconsistent lower endpoint")
        require(actual_upper == expected_upper, f"{path}: inconsistent upper endpoint")
        require(actual_lower <= actual_upper, f"{path}: reversed endpoints")
        count += 1
    require(count > 0, "no serialized ball records found")
    return count


def verify_hash_files(package_dir: Path, certificate: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    environment_path = package_dir / "environment.json"
    if environment_path.exists():
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        for filename, expected in environment.get("source_sha256", {}).items():
            path = package_dir / filename
            if not path.exists():
                path = Path(__file__).resolve().parent / filename
            require(path.exists(), f"missing source file recorded in environment.json: {filename}")
            require(sha256_file(path) == expected, f"source hash mismatch: {filename}")
        for key, filename in [
            ("certificate_sha256", "certificate.json"),
            ("continuation_steps_csv_sha256", "continuation_steps.csv"),
            ("endpoint_enclosures_sha256", "endpoint_enclosures.json"),
        ]:
            expected = environment.get(key)
            if expected:
                require(sha256_file(package_dir / filename) == expected, f"hash mismatch: {filename}")
        require(
            environment.get("libraries", {}).get("python-flint") == flint.__version__,
            "python-flint version differs from the archived environment",
        )
        require(
            environment.get("libraries", {}).get("FLINT")
            == getattr(flint, "__FLINT_VERSION__", "unknown"),
            "FLINT version differs from the archived environment",
        )
        messages.append("environment and archived source/data hashes agree")

    checksum_path = package_dir / "certificate.sha256"
    if checksum_path.exists():
        expected = checksum_path.read_text(encoding="ascii").split()[0]
        require(sha256_file(package_dir / "certificate.json") == expected, "certificate.sha256 mismatch")
        messages.append("certificate.sha256 agrees")
    return messages


def lower(record: dict[str, Any]) -> fmpq:
    return fmpq(record["lower_rational"])


def upper(record: dict[str, Any]) -> fmpq:
    return fmpq(record["upper_rational"])


def record_contains(outer: dict[str, Any], inner: dict[str, Any]) -> bool:
    return lower(outer) <= lower(inner) and upper(inner) <= upper(outer)


def verify_csv(package_dir: Path, certificate: dict[str, Any]) -> str:
    path = package_dir / "continuation_steps.csv"
    require(path.exists(), "continuation_steps.csv is missing")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected_count = sum(run["continuation_step_count"] for run in certificate["runs"].values())
    require(len(rows) == expected_count, "CSV row count does not match the certificate")
    by_key = {(row["run"], int(row["index"])): row for row in rows}
    require(len(by_key) == len(rows), "duplicate continuation rows in CSV")
    for run_name, run in certificate["runs"].items():
        for step in run["continuation_steps"]:
            row = by_key[(run_name, step["index"])]
            require(row["x0"] == step["x0"], f"{run_name} step {step['index']}: CSV x0 mismatch")
            require(row["h"] == step["h"], f"{run_name} step {step['index']}: CSV h mismatch")
            require(
                row["polynomial_sha256"] == step["polynomial_sha256"],
                f"{run_name} step {step['index']}: CSV polynomial hash mismatch",
            )
            require(
                fmpq(row["D_tube_lower"]) == lower(step["D_tube"]),
                f"{run_name} step {step['index']}: CSV D lower mismatch",
            )
    return f"continuation_steps.csv contains all {len(rows)} archived steps"


def verify_step_chain(certificate: dict[str, Any]) -> list[str]:
    config = certificate["configuration"]
    require(config == DEFAULT_CONFIG.as_dict(), "certificate configuration differs from this source version")
    local_h = decimal_to_fmpq(config["local_h"])
    max_step = decimal_to_fmpq(config["max_regular_step"])
    threshold = decimal_to_fmpq(D_DRAFT_THRESHOLD_DECIMAL)
    messages: list[str] = []

    for run_name, run in certificate["runs"].items():
        local = run["local_taylor_validation"]
        require(local["order_y"] == config["local_order"], f"{run_name}: local y order mismatch")
        require(local["order_f"] == config["local_order"] + 1, f"{run_name}: local f order mismatch")
        require(upper(local["contraction_q"]) < 1, f"{run_name}: local q is not below one")
        require(upper(local["tail_y_V"]) < decimal_to_fmpq(local["trial_radius"]), f"{run_name}: local tail exceeds tube")
        require(lower(local["D_tube"]) > 0, f"{run_name}: local D is not positive")

        steps = run["continuation_steps"]
        require(len(steps) == run["continuation_step_count"], f"{run_name}: step count mismatch")
        x = local_h
        global_D = run["D_global_tube"]
        require(record_contains(global_D, local["D_tube"]), f"{run_name}: global D does not contain local D")
        for expected_index, step in enumerate(steps):
            require(step["index"] == expected_index, f"{run_name}: nonsequential step index")
            require(fmpq(step["x0"]) == x, f"{run_name} step {expected_index}: broken x chain")
            expected_h = min(x / 4, max_step, 1 - x)
            require(fmpq(step["h"]) == expected_h, f"{run_name} step {expected_index}: step rule mismatch")
            require(fmpq(step["x1"]) == x + expected_h, f"{run_name} step {expected_index}: x1 mismatch")
            require(step["order_y"] == config["regular_order"], f"{run_name} step {expected_index}: y order mismatch")
            require(step["order_f"] == config["regular_order"] + 1, f"{run_name} step {expected_index}: f order mismatch")
            require(lower(step["residual_denominator_lower"]) > 0, f"{run_name} step {expected_index}: residual denominator nonpositive")
            require(lower(step["D_tube"]) > 0, f"{run_name} step {expected_index}: D nonpositive")
            require(
                upper(step["propagated_error_beta"]) < decimal_to_fmpq(step["trial_radius"]),
                f"{run_name} step {expected_index}: beta exceeds trial tube",
            )
            require(record_contains(global_D, step["D_tube"]), f"{run_name} step {expected_index}: global D hull mismatch")
            x += expected_h
        require(x == 1, f"{run_name}: continuation does not end at x=1")
        require(
            record_contains(run["endpoint_x1_s0"]["f"], steps[-1]["state_out"]["f"])
            and record_contains(steps[-1]["state_out"]["f"], run["endpoint_x1_s0"]["f"]),
            f"{run_name}: endpoint f differs from the last step",
        )
        require(
            record_contains(run["endpoint_x1_s0"]["y"], steps[-1]["state_out"]["y"])
            and record_contains(steps[-1]["state_out"]["y"], run["endpoint_x1_s0"]["y"]),
            f"{run_name}: endpoint y differs from the last step",
        )
        messages.append(f"{run_name}: local proof and {len(steps)}-step continuation chain are internally consistent")

    require(lower(certificate["runs"]["A_interval"]["D_global_tube"]) > threshold, "uniform D threshold failure")
    return messages


def compare_recomputed_branch(stored: dict[str, Any], recomputed: dict[str, Any], run_name: str) -> None:
    stored_local = stored["local_taylor_validation"]
    new_local = recomputed["local_taylor_validation"]
    require(stored_local["polynomial_sha256"] == new_local["polynomial_sha256"], f"{run_name}: local polynomial hash changed")
    for key in [
        "B_interval",
        "residual_numerator_bound",
        "residual_denominator_lower",
        "residual_bound_Rs",
        "M_f",
        "M_y_plus_one",
        "contraction_q",
        "tail_y_V",
        "total_f_error_at_h",
        "D_tube",
    ]:
        require(record_contains(stored_local[key], new_local[key]), f"{run_name}: recomputed local {key} is not archived")
    for component in ["f", "y"]:
        require(
            record_contains(stored_local["state_at_h"][component], new_local["state_at_h"][component]),
            f"{run_name}: recomputed local state {component} is not archived",
        )

    stored_steps = stored["continuation_steps"]
    new_steps = recomputed["continuation_steps"]
    require(len(stored_steps) == len(new_steps), f"{run_name}: recomputed step count changed")
    for old, new in zip(stored_steps, new_steps, strict=True):
        index = old["index"]
        for scalar_key in ["index", "x0", "h", "x1", "polynomial_sha256"]:
            require(old[scalar_key] == new[scalar_key], f"{run_name} step {index}: {scalar_key} changed")
        for ball_key in [
            "initial_error",
            "residual_numerator_bound",
            "residual_denominator_lower",
            "residual_bound_rho",
            "jacobian_bound_L",
            "propagated_error_beta",
            "D_tube",
        ]:
            require(record_contains(old[ball_key], new[ball_key]), f"{run_name} step {index}: recomputed {ball_key} is not archived")
        for group in ["p_end", "state_out"]:
            for component in ["f", "y"]:
                require(
                    record_contains(old[group][component], new[group][component]),
                    f"{run_name} step {index}: recomputed {group}.{component} is not archived",
                )

    for component in ["f", "y"]:
        require(
            record_contains(stored["endpoint_x1_s0"][component], recomputed["endpoint_x1_s0"][component]),
            f"{run_name}: recomputed endpoint {component} is not archived",
        )
    require(record_contains(stored["D_global_tube"], recomputed["D_global_tube"]), f"{run_name}: recomputed D hull is not archived")


def full_recompute(certificate: dict[str, Any]) -> list[str]:
    A_minus = decimal_to_fmpq(A_MINUS_DECIMAL)
    A_plus = decimal_to_fmpq(A_PLUS_DECIMAL)
    messages: list[str] = []

    def progress(message: str) -> None:
        print(message, flush=True)

    for label, low, high in [
        ("A_minus", A_minus, A_minus),
        ("A_plus", A_plus, A_plus),
        ("A_interval", A_minus, A_plus),
    ]:
        print(f"recomputing {label}", flush=True)
        record, endpoint, D_global = run_branch_validation(label, low, high, DEFAULT_CONFIG, progress)
        compare_recomputed_branch(certificate["runs"][label], record, label)
        if label == "A_minus":
            require(endpoint[0] > 0, "recomputed A_minus endpoint is not positive")
        elif label == "A_plus":
            require(endpoint[0] < 0, "recomputed A_plus endpoint is not negative")
        else:
            require(D_global > decimal_to_fmpq(D_DRAFT_THRESHOLD_DECIMAL), "recomputed uniform D bound failed")
        messages.append(f"{label}: full Arb recomputation is contained in the archive")
    return messages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory containing certificate.json and the source files",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="rerun all three Arb validations in addition to exact-rational checks",
    )
    args = parser.parse_args()
    package_dir = args.package_dir.resolve()
    certificate_path = package_dir / "certificate.json"
    require(certificate_path.exists(), f"missing {certificate_path}")
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))

    print(f"python-flint {flint.__version__}; FLINT {getattr(flint, '__FLINT_VERSION__', 'unknown')}")
    count = verify_ball_serialization(certificate)
    print(f"PASS: {count} ball records have consistent exact-rational serialization")
    for message in verify_hash_files(package_dir, certificate):
        print(f"PASS: {message}")
    for message in quick_certificate_checks(certificate):
        print(f"PASS: {message}")
    for message in verify_step_chain(certificate):
        print(f"PASS: {message}")
    print(f"PASS: {verify_csv(package_dir, certificate)}")

    if args.recompute:
        for message in full_recompute(certificate):
            print(f"PASS: {message}")

    print("CERTIFICATE VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
