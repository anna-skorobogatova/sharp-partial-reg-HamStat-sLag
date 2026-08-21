#!/bin/sh
set -eu
cd "$(dirname "$0")"
python verify_certificate.py
python verify_certificate.py --recompute
