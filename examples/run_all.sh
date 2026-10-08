#!/usr/bin/env bash
# Reproduce every committed example run. Usage: examples/run_all.sh
# The notebooks run on examples/.venv (pinned in requirements.txt); nb2p itself runs from the repo.
set -uo pipefail
cd "$(dirname "$0")"
NB2P=${NB2P:-"uv run --project .. nb2p"}
if [ ! -x .venv/bin/python ]; then
  uv venv .venv --python 3.12 && VIRTUAL_ENV=.venv uv pip install -r requirements.txt
fi

echo "== pandas-cookbook-snowiest-month"
cd pandas-cookbook-snowiest-month
$NB2P analyze snowiest_month.ipynb --out evidence/analysis.json
$NB2P capture snowiest_month.ipynb --out evidence/reference --repeat 2
$NB2P verify --pipeline snowiest/pipeline.py:run --reference evidence/reference --out evidence
$NB2P verify --pipeline mistakes.py:run_month_start --reference evidence/reference --out evidence/caught-mistake
DRAFT=$(mktemp -d)
$NB2P scaffold snowiest_month.ipynb --out "$DRAFT" --package snowiest_draft --reference evidence/reference
$NB2P verify --pipeline "$DRAFT/src/snowiest_draft/pipeline.py:run" --reference evidence/reference --out evidence/scaffold-draft
rm -rf "$DRAFT"
$NB2P report --reference evidence/reference --out evidence
cd ..

echo "== sklearn-feature-scaling"
cd sklearn-feature-scaling
$NB2P analyze plot_scaling_importance.ipynb --out evidence/analysis.json
$NB2P capture plot_scaling_importance.ipynb --out evidence/reference --repeat 2
$NB2P verify --pipeline scaling/pipeline.py:run --reference evidence/reference --out evidence
$NB2P verify --pipeline scaling/pipeline.py:run_fixed --reference evidence/reference --out evidence/intentional-fix
../.venv/bin/python -m scaling.pipeline | tee evidence/metrics.txt
../.venv/bin/python -m scaling.pipeline --fixed | tee evidence/intentional-fix/metrics.txt
DRAFT=$(mktemp -d)
$NB2P scaffold plot_scaling_importance.ipynb --out "$DRAFT" --package scaling_draft --reference evidence/reference
$NB2P verify --pipeline "$DRAFT/src/scaling_draft/pipeline.py:run" --reference evidence/reference --out evidence/scaffold-draft
rm -rf "$DRAFT"
$NB2P report --reference evidence/reference --out evidence
cd ..

echo "== broken-hidden-state"
cd broken-hidden-state
$NB2P analyze iris_session.ipynb --out evidence/analysis.json
$NB2P capture iris_session.ipynb --out evidence/reference
$NB2P verify --pipeline pipeline.py:run --reference evidence/reference --out evidence
$NB2P report --reference evidence/reference --out evidence
cd ..
