#!/usr/bin/env bash
# Separate CPU environment; does not download models or execute inference.
set -euo pipefail

answerability_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
answerability_env="${1:-${answerability_root}/../.venv-admission}"
answerability_python="${PYTHON:-python3.12}"

command -v uv >/dev/null || { printf 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/\n' >&2; exit 127; }
uv venv --python "${answerability_python}" --allow-existing "${answerability_env}"
uv pip install --python "${answerability_env}/bin/python" --torch-backend cpu -r "${answerability_root}/requirements/answerability.txt"
uv pip install --python "${answerability_env}/bin/python" --torch-backend cpu -c "${answerability_root}/requirements/answerability.txt" -e "${answerability_root}[mcp,pdf,vision]"
uv pip check --python "${answerability_env}/bin/python"
