#!/usr/bin/env bash
# Build .venv-xai: the overlay environment the OmniXAI analysis runs in.
#
# --system-site-packages is the point of it. numpy, pandas, scikit-learn and the
# rest are inherited from the environment the models were TRAINED in, so a
# joblib artefact loads unchanged; only the packages that omnixai pins
# differently (dash, Flask, shap) are shadowed inside the venv. Nothing outside
# .venv-xai is modified.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${XAI_VENV:-$ROOT/.venv-xai}"
PYTHON="${PYTHON:-python3}"

echo "==> creating $VENV"
"$PYTHON" -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip

echo "==> installing omnixai (no deps: its pins predate this repo's stack)"
"$VENV/bin/python" -m pip install --quiet --no-deps "omnixai==1.3.1"

echo "==> installing the pinned explainability stack"
"$VENV/bin/python" -m pip install --quiet -r "$ROOT/requirements-xai.txt"

# hnswlib has no wheel, so pip builds it and drags the newest numpy in as a
# dependency. That numpy would shadow the one the models were fitted under -
# and numba (which shap needs) refuses to import against it. Drop it, but only
# if it really landed inside the venv.
if "$VENV/bin/python" -c "import numpy, sys; sys.exit(0 if numpy.__file__.startswith('$VENV') else 1)"; then
    echo "==> dropping the venv numpy override, back to the system one"
    "$VENV/bin/python" -m pip uninstall --quiet -y numpy
fi

"$VENV/bin/python" - <<'PY'
import numpy, sklearn, shap, dash, hnswlib, omnixai
from omnixai.explainers.tabular import TabularExplainer
from omnixai.visualization.dashboard import Dashboard
print(f"ready: omnixai {omnixai.__version__}, shap {shap.__version__}, "
      f"dash {dash.__version__}, numpy {numpy.__version__}, sklearn {sklearn.__version__}")
PY
echo "==> done. Run: make xai"
