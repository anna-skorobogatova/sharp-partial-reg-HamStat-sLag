# Independent audit package

This directory contains four code paths that are separate from the primary
`validator_core.py` implementation.

1. `independent_symbolic_audit.py` starts from the original Cartan cubic,
   constructs the full 5-by-5 Hessian with SymPy, and verifies the reduction,
   the printed D/N polynomials, the endpoint identity, and the corrected
   determinant formula as exact identities.
2. `independent_arb_validator.py` treats the archived rational Taylor
   polynomials as proposed witnesses, recomputes all a-posteriori inequalities
   from the complex K,T formulas at 768 bits, and uses 128/32 path cells.
3. `independent_decimal_validator.py` has no Arb, FLINT, or SymPy dependency.
   It uses standard-library Decimal intervals with explicit directed rounding
   at 160 digits and recomputes the local and regular proof inequalities,
   endpoint signs, and simultaneous parameter D bound.
4. `independent_rational_validation.py` generates its own Taylor models and
   uses exact Fraction interval inequalities, a singular start x=1/250,
   singular order 24, regular order 26, and the different rule
   h=min(x/3,1/100,1-x). It certifies the two point-parameter endpoint signs.

The included `certificate.json` and witness polynomial file are data, not trusted certificates: the Arb and
Decimal checkers prove the required residual and tube inequalities afresh.
The rational checker does not use this file.

## Commands

From this directory:

```bash
python independent_symbolic_audit.py
python independent_arb_validator.py
python independent_decimal_validator.py
python independent_rational_validation.py independent_rational_results.json
```

The Decimal and rational scripts use only the Python standard library. The
corresponding logs and JSON results included here were produced on CPython
3.13.5. The symbolic log used SymPy 1.14.0. The Arb log used python-flint 0.9.0
with FLINT 3.6.0.

The clean rerun of the primary implementation is not called an independent
audit. No external human code review is claimed by these files.
