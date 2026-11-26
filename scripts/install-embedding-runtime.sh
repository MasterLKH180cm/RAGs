#!/usr/bin/env bash
# Isolated CPU environment; model downloads require a later explicit invocation.
set -euo pipefail

embedding_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
embedding_env="${1:-${embedding_root}/../.venv-embedding}"
embedding_python="${PYTHON:-python3.12}"

command -v uv >/dev/null || { printf 'Install uv: https://docs.astral.sh/uv/\n' >&2; exit 127; }
uv venv --python "${embedding_python}" --allow-existing "${embedding_env}"
uv pip install --python "${embedding_env}/bin/python" --torch-backend cpu -r "${embedding_root}/requirements/embedding-runtime.txt"
uv pip install --python "${embedding_env}/bin/python" --torch-backend cpu \
  -c "${embedding_root}/requirements/embedding-runtime.txt" \
  -e "${embedding_root}[dev,encoders,tokenizers,sentence-transformers]"
uv pip check --python "${embedding_env}/bin/python"
