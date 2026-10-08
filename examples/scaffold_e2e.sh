#!/usr/bin/env bash
# End-to-end check of what `nb2p scaffold` generates, in fresh projects outside this repository.
#
# For each case: a new project directory with one example notebook, `capture`, `scaffold`, the
# install command the scaffold prints, then the generated test (must pass), the same test after
# one deliberate change to a pipeline output (must fail), and finally the `run:` steps of the
# generated GitHub Actions workflow in a clean copy of the project (must pass).
#
# Usage:
#   examples/scaffold_e2e.sh                     notebook-to-pipeline==<pyproject version> from PyPI
#   FIND_LINKS=dist examples/scaffold_e2e.sh     the same version from a local build (uv build)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VERSION=${VERSION:-$(grep -m1 '^version' "$HERE/../pyproject.toml" | cut -d'"' -f2)}
WORK=$(mktemp -d)
if [ -n "${FIND_LINKS:-}" ]; then
  UV_FIND_LINKS=$(cd "$FIND_LINKS" && pwd)
  export UV_FIND_LINKS
  # uv treats a wheel with the same name and version as unchanged, so a rebuilt local wheel needs
  # an empty cache to be picked up.
  export UV_CACHE_DIR=$WORK/uv-cache
fi
# Re-check notebook-to-pipeline itself on the index, never an older cached copy of it.
export UV_REFRESH_PACKAGE=notebook-to-pipeline
NB2P="uvx notebook-to-pipeline==$VERSION"
PY312=$(cd / && uv python find --no-project 3.12)  # a CPython 3.12 outside any project venv
SYS_PATH=/usr/bin:/bin         # PATH without uv, as on a machine that never installed it
RESULTS=()
unset VIRTUAL_ENV CI

note() { printf '\n== %s\n' "$*"; }
run() { printf '$ %s\n' "$*"; "$@"; }
record() { RESULTS+=("$1: $2"); printf '>> %s: %s\n' "$1" "$2"; }

expect() {  # expect <case> <label> <pass|fail> <command...>
  local case=$1 label=$2 want=$3
  shift 3
  local out rc
  out=$("$@" 2>&1)
  rc=$?
  printf '%s\n' "$out" | tail -n 25
  if { [ "$want" = pass ] && [ $rc -eq 0 ]; } || { [ "$want" = fail ] && [ $rc -ne 0 ]; }; then
    record "$case" "$label: exit $rc as expected ($want)"
  else
    record "$case" "$label: UNEXPECTED exit $rc (wanted $want)"
  fi
}

ci_steps() {  # the `run:` lines of the generated workflow, in order
  sed -n 's/^ *- run: //p' .github/workflows/equivalence.yml
}

ci_python() {
  sed -n 's/^ *python-version: "\(.*\)"/\1/p' .github/workflows/equivalence.yml | head -n1
}

run_ci_copy() {  # run_ci_copy <case> <kind>: the workflow's steps in a clean copy of the project
  local case=$1 kind=$2 src=$PWD dst=$WORK/$1-ci
  mkdir -p "$dst"
  (cd "$src" && tar cf - --exclude .venv --exclude .nb2p --exclude .nb2p-verify \
    --exclude .pytest_cache --exclude __pycache__ .) | (cd "$dst" && tar xf -)
  cd "$dst" || return
  note "$case: generated workflow, clean copy in $dst"
  cat .github/workflows/equivalence.yml
  local series
  series=$(ci_python)
  local script
  script=$(ci_steps)
  if [ "$kind" = uv ]; then
    # astral-sh/setup-uv with python-version sets UV_PYTHON.
    expect "$case" "CI steps" pass env CI=true UV_PYTHON="$series" bash -ec "$script"
  else
    # actions/setup-python puts that Python first on PATH; nothing else (no uv) is installed.
    "$(cd / && uv python find --no-project "$series")" -m venv "$WORK/$case-setup-python"
    expect "$case" "CI steps" pass env CI=true PATH="$WORK/$case-setup-python/bin:$SYS_PATH" \
      bash -ec "$script"
  fi
  cd "$src" || return
}

copy_notebook() {  # copy_notebook <example dir> <notebook> <dest>
  mkdir -p "$3"
  cp "$HERE/$1/$2" "$3/"
  if [ -d "$HERE/$1/data" ]; then cp -R "$HERE/$1/data" "$3/"; fi
}

mutate() {  # mutate <file glob dir> <sed expression>: change one pipeline output on purpose
  local f
  f=$(grep -l "$2" src/*/*.py | head -n1)
  printf 'deliberate change in %s: %s -> %s\n' "$f" "$2" "$3"
  cp "$f" "$f.orig"
  sed -i.bak "s/$2/$3/" "$f" && rm -f "$f.bak"
}

restore() {
  local f
  for f in src/*/*.py.orig; do mv "$f" "${f%.orig}"; done
}

echo "notebook-to-pipeline $VERSION${UV_FIND_LINKS:+ (from $UV_FIND_LINKS)}, work dir $WORK"
uv --version

# ---------------------------------------------------------------------------------------------
note "uv-new: sklearn notebook in an empty folder, following the README quickstart"
P=$WORK/uv-new
copy_notebook sklearn-feature-scaling plot_scaling_importance.ipynb "$P"
cd "$P" || exit 1
run uvx --with scikit-learn,pandas,matplotlib "notebook-to-pipeline==$VERSION" capture \
  plot_scaling_importance.ipynb > capture.log 2>&1
grep -E "^(Python|Top-to-bottom|Captured|Figures|Printed)" capture.log
run $NB2P scaffold plot_scaling_importance.ipynb --out .
expect uv-new "install (uv sync)" pass uv sync
expect uv-new "generated test" pass uv run pytest -q
mutate x "random_state=42" "random_state=7"
expect uv-new "generated test after a deliberate change" fail uv run pytest -q
restore
run_ci_copy uv-new uv

# ---------------------------------------------------------------------------------------------
note "uv-existing: pandas-cookbook notebook in an existing uv project (pyproject.toml, uv.lock)"
P=$WORK/uv-existing
copy_notebook pandas-cookbook-snowiest-month snowiest_month.ipynb "$P"
cd "$P" || exit 1
run uv init --bare --python 3.12 -q
run uv add -q pandas==2.2.3 numpy==2.1.2 matplotlib==3.9.2 ipykernel
run $NB2P capture snowiest_month.ipynb > capture.log 2>&1
grep -E "^(Python|Top-to-bottom|Captured|Figures|Printed)" capture.log
run $NB2P scaffold snowiest_month.ipynb --out .
expect uv-existing "install (uv add --dev pytest, as the scaffold says)" pass uv add -q --dev pytest
expect uv-existing "generated test" pass uv run pytest -q
mutate x "resample('M').apply(np.median)$" "resample('M').apply(np.mean)"
expect uv-existing "generated test after a deliberate change" fail uv run pytest -q
restore
run_ci_copy uv-existing uv

# ---------------------------------------------------------------------------------------------
for EX in sklearn snowiest; do
  if [ $EX = sklearn ]; then
    DIR=sklearn-feature-scaling NB=plot_scaling_importance.ipynb
    FROM="random_state=42" TO="random_state=7"
  else
    DIR=pandas-cookbook-snowiest-month NB=snowiest_month.ipynb
    FROM="resample('M').apply(np.median)$" TO="resample('M').apply(np.mean)"
  fi
  CASE=pip-$EX
  note "$CASE: plain venv + pip, no uv on PATH for anything that runs inside the project"
  P=$WORK/$CASE
  copy_notebook $DIR $NB "$P"
  cd "$P" || exit 1
  run "$PY312" -m venv .venv
  run .venv/bin/python -m pip install -q -r "$HERE/requirements.txt"
  run $NB2P capture $NB > capture.log 2>&1
  grep -E "^(Python|Top-to-bottom|Captured|Figures|Printed)" capture.log
  run $NB2P scaffold $NB --out .
  expect $CASE "install (pip install -r requirements-dev.txt)" pass \
    .venv/bin/python -m pip install -q -r requirements-dev.txt
  expect $CASE "generated test" pass env PATH="$P/.venv/bin:$SYS_PATH" python -m pytest -q
  mutate x "$FROM" "$TO"
  expect $CASE "generated test after a deliberate change" fail \
    env PATH="$P/.venv/bin:$SYS_PATH" python -m pytest -q
  restore
  run_ci_copy $CASE pip
done

note "summary"
printf '%s\n' "${RESULTS[@]}"
if printf '%s\n' "${RESULTS[@]}" | grep -q UNEXPECTED; then
  rm -rf "$WORK/uv-cache"
  echo "kept $WORK for inspection"
  exit 1
fi
rm -rf "$WORK"
