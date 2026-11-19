#!/usr/bin/env bash
# Build .venv-xai: the overlay environment the OmniXAI analysis runs in.
#
# The point of it: numpy, pandas, scikit-learn and the rest are inherited from the
# environment the models were TRAINED in, so a joblib artefact loads unchanged; only
# the packages that omnixai pins differently (dash, Flask, shap) are shadowed inside
# the venv. Nothing outside .venv-xai is modified.
#
# Inheritance: when the repo's own .venv exists (the training environment), the
# overlay reads its site-packages through a .pth file, after its own; otherwise it
# falls back to --system-site-packages. Works on Linux, macOS and Windows (Git Bash).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${XAI_VENV:-$ROOT/.venv-xai}"
BASE_VENV="${XAI_BASE_VENV:-$ROOT/.venv}"

venv_python() {            # a venv's interpreter, wherever the platform puts it
    if [ -x "$1/Scripts/python.exe" ]; then echo "$1/Scripts/python.exe"; else echo "$1/bin/python"; fi
}

if [ -d "$BASE_VENV" ]; then
    BASE_PY="$(venv_python "$BASE_VENV")"
    PYTHON="${PYTHON:-$BASE_PY}"
    echo "==> creating $VENV over the training environment $BASE_VENV"
    "$PYTHON" -m venv "$VENV"
    VPY="$(venv_python "$VENV")"
    BASE_SITE="$("$BASE_PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")"
    OWN_SITE="$("$VPY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")"
    echo "$BASE_SITE" > "$OWN_SITE/aavas_training_env.pth"
else
    PYTHON="${PYTHON:-python3}"
    echo "==> creating $VENV with the system site-packages"
    "$PYTHON" -m venv --system-site-packages "$VENV"
    VPY="$(venv_python "$VENV")"
fi
"$VPY" -m pip install --quiet --upgrade pip

echo "==> installing omnixai (no deps: its pins predate this repo's stack)"
"$VPY" -m pip install --quiet --no-deps "omnixai==1.3.1"

echo "==> installing the pinned explainability stack"
if [ -x "$VENV/Scripts/python.exe" ]; then
    # hnswlib publishes no Windows wheel and needs the MSVC build tools to compile.
    # chroma-hnswlib is the same library (import name hnswlib) with prebuilt wheels;
    # --no-deps so it cannot pull in a newer numpy.
    REQS="$(mktemp)"
    grep -v -E '^hnswlib\b' "$ROOT/requirements-xai.txt" > "$REQS"
    "$VPY" -m pip install --quiet -r "$REQS"
    "$VPY" -m pip install --quiet --no-deps --only-binary=:all: chroma-hnswlib
    rm -f "$REQS"
else
    "$VPY" -m pip install --quiet -r "$ROOT/requirements-xai.txt"
fi

# hnswlib has no wheel, so pip builds it and drags the newest numpy in as a
# dependency. That numpy would shadow the one the models were fitted under -
# and numba (which shap needs) refuses to import against it. Drop it, but only
# if it really landed inside the venv.
if "$VPY" -c "import numpy, os, sys; sys.exit(0 if os.path.abspath(numpy.__file__).lower().startswith(os.path.abspath(r'$VENV').lower()) else 1)"; then
    echo "==> dropping the venv numpy override, back to the training environment's"
    "$VPY" -m pip uninstall --quiet -y numpy
fi

"$VPY" - <<'PY'
import numpy, sklearn, shap, dash, hnswlib, omnixai
from omnixai.explainers.tabular import TabularExplainer
from omnixai.visualization.dashboard import Dashboard
print(f"ready: omnixai {omnixai.__version__}, shap {shap.__version__}, "
      f"dash {dash.__version__}, numpy {numpy.__version__}, sklearn {sklearn.__version__}")
PY
echo "==> done. Run: make xai"
