# Computer-assisted proof code: Cartan-cubic special-Lagrangian shooting

## Primary proof

The `primary/` directory is the exact archived validation package used in the paper.

- `validator_core.py`: rigorous Arb interval implementation of the reduced ODE,
  singular Taylor contraction, regular Taylor/Gronwall continuation, and certificate checks.
- `generate_certificate.py`: executes the three proof runs (`A_minus`, `A_plus`, and
  the whole parameter interval) and writes the JSON/CSV certificate.
- `verify_certificate.py`: checks the archived certificate; `--recompute` reruns every
  Arb proof step from the source equations.
- `certificate.json`: full exact-rational machine-readable certificate.
- `continuation_steps.csv`: all 309 accepted continuation steps.
- `endpoint_enclosures.json`: compact endpoint-sign and global-D enclosure.
- `METHOD.md`: mathematical proof obligations implemented by the code.
- `requirements.txt`, `Dockerfile`, `run_validation.sh`: pinned execution environment.

Run:

```bash
cd primary
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python verify_certificate.py
python verify_certificate.py --recompute
```

Or:

```bash
cd primary
docker build -t cartan-sl-validator .
docker run --rm cartan-sl-validator
```

## Independent audits

The `independent_audits/` directory contains separately written symbolic, Arb,
directed-decimal, and exact rational-interval validators, together with their logs
and machine-readable outputs.

## Local exact checker

The `verification/` directory contains the exact-rational certificate audit used in
this workspace. Its archived script has an environment-specific `PKG` path; point it
to `primary/` before running outside the original workspace.
