#!/usr/bin/env bash
# Separate CPU environment; does not download models or execute inference.
set -euo pipefail

answerability_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
answerability_env="${1:-${answerability_root}/../.venv-admission}"
answerability_python="${PYTHON:-python3.12}"

"${answerability_python}" -m venv "${answerability_env}"
"${answerability_env}/bin/python" -m pip install 'torch==2.15.0+cpu' --index-url https://download.pytorch.org/whl/cpu
"${answerability_env}/bin/python" -m pip install -r "${answerability_root}/requirements/answerability.txt"
"${answerability_env}/bin/python" -m pip install -c "${answerability_root}/requirements/answerability.txt" -e "${answerability_root}[mcp,pdf,vision]"
