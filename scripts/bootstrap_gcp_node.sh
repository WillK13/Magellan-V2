#!/usr/bin/env bash
set -euo pipefail

NODE_ID="${1:-${MAGELLAN_NODE_ID:-}}"
REPO_ROOT="${MAGELLAN_REPOSITORY_ROOT:-$HOME/Magellan-V2}"
BRANCH="${MAGELLAN_BRANCH:-seven-node-deployment}"
RUN_TESTS="${MAGELLAN_BOOTSTRAP_RUN_TESTS:-1}"
VALIDATE_DATASETS="${MAGELLAN_BOOTSTRAP_VALIDATE_DATASETS:-1}"
PYTHON_BIN="${MAGELLAN_PYTHON_BIN:-python3.11}"

if [[ -z "$NODE_ID" ]]; then
  echo "usage: $0 <node-id>" >&2
  exit 2
fi

cd "$REPO_ROOT"

git fetch origin
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git switch "$BRANCH"
  git branch --set-upstream-to="origin/$BRANCH" "$BRANCH" 2>/dev/null || true
  git pull --ff-only "origin" "$BRANCH"
else
  git switch -c "$BRANCH" --track "origin/$BRANCH"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "Python 3.11+ is required but no Python 3 executable was found" >&2
    exit 1
  fi
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(
        f"Python 3.11+ required, found {sys.version.split()[0]}"
    )
print("python", sys.version.split()[0])
PY

for command in git rsync ssh curl; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "required command is missing: $command" >&2
    exit 1
  }
done

# Stage 4 uses all seven workers as valid MPI/Dendro destinations.  Keep the
# runtime preinstalled rather than folding package installation into migration
# time, which would contaminate the migration measurements.
if ! command -v mpirun >/dev/null 2>&1 || ! command -v mpiexec >/dev/null 2>&1; then
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "OpenMPI 4.1.4 is required, but apt-get is unavailable" >&2
    exit 1
  fi
  sudo -n apt-get update
  sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y \
    openmpi-bin libopenmpi-dev
fi

OPENMPI_BANNER="$(mpirun --version | head -n 1)"
if [[ ! "$OPENMPI_BANNER" =~ 4\.1\.4([^0-9]|$) ]]; then
  echo "OpenMPI 4.1.4 required; observed: $OPENMPI_BANNER" >&2
  exit 1
fi
echo "openmpi $OPENMPI_BANNER"

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

python - scripts/../config/cluster.gcp.json "$NODE_ID" <<'PY'
import sys
from magellan.config.loader import load_cluster_config

cluster = load_cluster_config(sys.argv[1])
node = cluster.get_node(sys.argv[2])
print(
    f"configured node={node.id} vm={node.vm_name} zone={node.zone} "
    f"ip={node.internal_ip} carbon={node.carbon_region}"
)
PY

if [[ "$RUN_TESTS" == "1" ]]; then
  ruff check magellan tests scripts
  python -m compileall -q magellan scripts
  python -m pytest -q
fi

if [[ "$VALIDATE_DATASETS" == "1" ]]; then
  python scripts/validate_seven_node_deployment.py
fi

git restore magellan_v2.egg-info 2>/dev/null || true

echo "BOOTSTRAP PASSED node=$NODE_ID branch=$BRANCH commit=$(git rev-parse HEAD)"
