# Validated shooting certificate for the Cartan-cubic special-Lagrangian example

This package supplies the missing computer-assisted part of the draft
construction: a validated singular Taylor start, two rigorous endpoint-sign
computations, and a simultaneous continuation tube for all
`A in [A_minus,A_plus]` with a uniform positive lower bound for `D`.

It is deliberately scoped to the shooting proposition. The symbolic Hessian
reduction, phase-branch argument, `C^{1,1}` regularity, failure of `C^2`, and
viscosity passage are separate analytic parts of the proof.

## Certified statements

The exact input parameters are

```text
A_minus = 0.66922990609204402834133929821078023454
A_plus  = 0.66922990609204402834133931821078023454
```

The archived Arb computation proves

```text
f_{A_minus}(0) in
[ 8.8552416517743152487231245241442132886508533093616e-27,
  8.8552418779480232407604677123569467798944658197284e-27 ]

f_{A_plus}(0) in
[-8.8552418383799560878257630888697631462027284258737e-27,
 -8.8552416122062480957884199006570296549591159155068e-27 ]
```

so the signs are strictly opposite. The simultaneous parameter run proves

```text
D(s,f_A(s),f_A'(s)) >=
4.6064916663925967921944345817214537636010522654540
```

for every `A in [A_minus,A_plus]` and every `0 <= s <= 1`. In particular this
strictly exceeds the draft threshold `4.2964512605`.

The exact rational endpoints of these balls—not merely rounded displays—are in
`endpoint_enclosures.json` and `certificate.json`.

Arb’s compact printer renders the wide global `D` hull in the raw log as
`[+/- 12.0]`; this is only a display abbreviation. The serialized lower
endpoint is the strictly positive number printed above.

## Reproducibility parameters

| Item | Archived value |
|---|---:|
| Python | CPython 3.13.5 |
| Interval library | python-flint 0.9.0 |
| FLINT / Arb backend | FLINT 3.6.0 |
| Working precision | 512 bits (about 154 decimal digits) |
| Singular interval | `0 <= x=1-s <= 0.01` |
| Singular Taylor order | `y`: 40, `f`: 41 |
| Singular subdivisions | 64 |
| Singular trial radius | `1e-12` |
| Regular Taylor order | `y`: 30, `f`: 31 |
| Regular step rule | `h=min(x/4,0.01,1-x)` |
| Regular subdivisions per step | 16 |
| Regular trial radius | `1e-10` |
| Continuation steps | 103 per run; 309 archived rows total |
| Rationalized state-center digits | 115 |
| Rationalized Taylor-coefficient digits | 130 |

The local contraction constant is

```text
q <= 0.19140324762136517122712575598335797390126711045738 < 1.
```

For the two point-parameter runs, the validated Taylor tails on `0<=x<=0.01`
are

```text
A_minus: |y-p_y| <= 6.9259337007214316585322385192581e-46
         |f-p_f| <= 6.9259337007214316585322385192581e-48

A_plus:  |y-p_y| <= 6.9259336999379129723447073113260e-46
         |f-p_f| <= 6.9259336999379129723447073113260e-48
```

For the simultaneous parameter run, the corresponding bounds are

```text
|y-p_y| <= 5.4104695341312586970528914545552e-26
|f-p_f| <= 1.0541046953413125869705289145456e-26.
```

The latter bounds include the full parameter radius `|A-A_center|<=1e-26`, so
they are intentionally much wider than the pure point-parameter truncation
tails. The common local residual bound is approximately `5.600288e-46`.

## Files

- `validator_core.py` — equations, exact-rational Taylor centers, singular
  contraction proof, residual bounds, Gronwall continuation, and interval
  tube checks.
- `generate_certificate.py` — runs the three validations and writes the JSON
  and CSV archive.
- `verify_certificate.py` — exact-rational consistency checker; with
  `--recompute`, reruns every Arb proof step.
- `certificate.json` — full machine-readable certificate, including local
  Taylor coefficients and every continuation enclosure.
- `endpoint_enclosures.json` — compact endpoint-sign and global-`D` summary.
- `continuation_steps.csv` — one row for each of the 309 regular continuation
  steps, including residual, Jacobian, propagated error, state, and `D` data.
- `environment.json` — interpreter/library versions and source/data hashes.
- `METHOD.md` — mathematical proof obligations implemented by the code.
- `requirements.txt`, `Dockerfile`, and `run_validation.sh` — pinned execution
  environment and convenience entry points.

## Verification

Create an isolated environment and run the exact-rational checker:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python verify_certificate.py
```

The full independent recomputation is:

```bash
python verify_certificate.py --recompute
```

To regenerate the archive from the equations:

```bash
python generate_certificate.py --output-dir regenerated
python verify_certificate.py --package-dir regenerated
```

The Docker entry point performs the full recomputation:

```bash
docker build -t cartan-sl-validator .
docker run --rm cartan-sl-validator
```

## What the checker establishes

The quick checker verifies all ball encodings with exact rational arithmetic,
all archived hashes, the local contraction inequalities, the exact continuation
step chain, every `beta<trial_radius` inequality, every positive `D` lower
bound, the endpoint signs, and consistency of all 309 CSV rows.

The `--recompute` mode additionally rebuilds every rational Taylor polynomial,
residual bound, Jacobian bound, tube, outgoing state, endpoint enclosure, and
global `D` hull with Arb. It checks that the recomputed results are contained
in the archived exact-rational enclosures and that all polynomial hashes match.

The numerical rigor rests on Arb's outward-rounded ball arithmetic and on the
singular contraction and regular residual/Gronwall estimates set out in
`METHOD.md`. As with any computer-assisted proof, independent source review and
re-execution on another machine remain appropriate before publication.
