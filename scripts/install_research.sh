#!/usr/bin/env bash
# Reproduce one pinned research engine without changing the core environment.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 || ! "$1" =~ ^(raptor|lightrag)$ ]]; then
  printf 'Usage: bash scripts/install_research.sh raptor|lightrag VENV_DIR [RAPTOR_SOURCE_DIR]\n' >&2
  exit 2
fi
research_engine="$1"
research_venv="$2"
research_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
research_python="${RAGS_RESEARCH_BOOTSTRAP_PYTHON:-python3.12}"
command -v uv >/dev/null || { printf 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/\n' >&2; exit 127; }
# Keep pip available for upstream pipmaster compatibility; installation uses uv.
uv venv --python "$research_python" --seed --allow-existing "$research_venv"
research_venv="$(cd "$research_venv" && pwd)"
uv pip install --python "$research_venv/bin/python" --torch-backend cpu -r "$research_repo/requirements/research-$research_engine.txt"
uv pip install --python "$research_venv/bin/python" --no-deps -e "$research_repo"
if [[ "$research_engine" == raptor ]]; then
  research_source="${3:-$research_venv/upstream-raptor}"
  research_commit="7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767"
  if [[ ! -e "$research_source" ]]; then
    git clone --no-checkout https://github.com/parthsarthi03/raptor.git "$research_source"
    git -C "$research_source" checkout --detach "$research_commit"
  fi
  if [[ "$(git -C "$research_source" rev-parse HEAD)" != "$research_commit" ]] ||
     [[ -n "$(git -C "$research_source" status --porcelain --untracked-files=no)" ]]; then
    printf 'RAPTOR checkout must be the clean pinned source revision.\n' >&2
    exit 2
  fi
  research_source="$(cd "$research_source" && pwd)"
  "$research_venv/bin/python" - "$research_source" <<'PY'
from pathlib import Path
import site
import sys
Path(site.getsitepackages()[0], "rags_raptor_upstream.pth").write_text(sys.argv[1] + "\n")
PY
fi
uv pip check --python "$research_venv/bin/python"
