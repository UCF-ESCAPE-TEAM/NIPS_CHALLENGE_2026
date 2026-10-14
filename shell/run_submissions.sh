#!/usr/bin/env bash
# run_submissions.sh
#
# Test and package every model submission for the NeurIPS 2026 EEG/EMG
# Foundation Challenge (Track 4, EMG-to-Pose) with the official benchopt
# harness from the 2026-competition repository.
#
# Layout expected under SUBMISSION_ROOT (one sub-folder per model):
#   SUBMISSION_ROOT/
#     cnn1d_pose/   submission.py  weights.pt
#     ulhm_pose/    submission.py  weights.pt
#
# For each sub-folder the script:
#   1. checks that submission.py and the weights file exist
#   2. reads the Solver.name from submission.py
#   3. copies submission.py into <COMPETITION>/tracks/emg_pose/solvers/<folder>.py
#   4. runs:  COMPET_SUBMISSION_DIR=<folder> benchopt run tracks/emg_pose -d Simulated -s <Solver.name>
#   5. zips the folder flat into SUBMISSION_ROOT/zips/<folder>.zip (ready for Codabench)
#   6. writes a log to SUBMISSION_ROOT/logs/<folder>.log and prints a summary table
#
# Usage:
#   bash run_submissions.sh SUBMISSION_ROOT COMPETITION_DIR [options]
#
# Options:
#   --only NAME[,NAME...]  only process these sub-folders
#   --dataset NAME         benchopt dataset (default: Simulated)
#   --track NAME           track folder under <COMPETITION>/tracks (default: emg_pose)
#   --weights FILE         weights file name expected in each folder (default: weights.pt)
#   --no-zip               skip creating the ZIPs
#   --zip-only             only create the ZIPs, do not run benchopt
#   --keep-solvers         leave the copied solver files in tracks/<track>/solvers/
#                          (default: remove them after the run)
#   --dry-run              print what would be done, run nothing
#   -h, --help             show this help
#
# Requirements: bash, zip, python3, benchopt (installed in the active env).
# Exit code: 0 when every processed model passed, 1 otherwise.

set -uo pipefail

# ------------------------------------------------------------------ arguments
SUB_ROOT=""; COMP_DIR=""
ONLY=""; DATASET="Simulated"; TRACK="emg_pose"; WEIGHTS="weights.pt"
DO_ZIP=1; DO_RUN=1; KEEP_SOLVERS=0; DRY_RUN=0

usage() { sed -n '2,38p' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)         ONLY="$2"; shift 2 ;;
    --only=*)       ONLY="${1#*=}"; shift ;;
    --dataset)      DATASET="$2"; shift 2 ;;
    --dataset=*)    DATASET="${1#*=}"; shift ;;
    --track)        TRACK="$2"; shift 2 ;;
    --track=*)      TRACK="${1#*=}"; shift ;;
    --weights)      WEIGHTS="$2"; shift 2 ;;
    --weights=*)    WEIGHTS="${1#*=}"; shift ;;
    --no-zip)       DO_ZIP=0; shift ;;
    --zip-only)     DO_RUN=0; shift ;;
    --keep-solvers) KEEP_SOLVERS=1; shift ;;
    --dry-run)      DRY_RUN=1; shift ;;
    -h|--help)      usage; exit 0 ;;
    -*)             echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [[ -z "$SUB_ROOT" ]]; then SUB_ROOT="$1"
      elif [[ -z "$COMP_DIR" ]]; then COMP_DIR="$1"
      else echo "unexpected argument: $1" >&2; exit 2; fi
      shift ;;
  esac
done

if [[ -z "$SUB_ROOT" || -z "$COMP_DIR" ]]; then
  echo "error: SUBMISSION_ROOT and COMPETITION_DIR are required" >&2; usage >&2; exit 2
fi
SUB_ROOT=$(cd "$SUB_ROOT" 2>/dev/null && pwd) || { echo "not a directory: $SUB_ROOT" >&2; exit 2; }
COMP_DIR=$(cd "$COMP_DIR" 2>/dev/null && pwd) || { echo "not a directory: $COMP_DIR" >&2; exit 2; }

TRACK_DIR="$COMP_DIR/tracks/$TRACK"
SOLVERS_DIR="$TRACK_DIR/solvers"
[[ -d "$SOLVERS_DIR" ]] || { echo "solvers folder not found: $SOLVERS_DIR" >&2; exit 2; }

for tool in python3; do command -v "$tool" >/dev/null || { echo "$tool is required" >&2; exit 2; }; done
if [[ $DO_ZIP -eq 1 ]]; then command -v zip >/dev/null || { echo "zip is required (or use --no-zip)" >&2; exit 2; }; fi
if [[ $DO_RUN -eq 1 && $DRY_RUN -eq 0 ]]; then
  command -v benchopt >/dev/null || { echo "benchopt not found in PATH; activate the competition environment" >&2; exit 2; }
fi

LOG_DIR="$SUB_ROOT/logs"; ZIP_DIR="$SUB_ROOT/zips"
[[ $DRY_RUN -eq 1 ]] || mkdir -p "$LOG_DIR" "$ZIP_DIR"

echo "Submission root : $SUB_ROOT"
echo "Competition dir : $COMP_DIR"
echo "Track / dataset : $TRACK / $DATASET"
echo

# ------------------------------------------------------------ model folders
MODELS=()
if [[ -n "$ONLY" ]]; then
  IFS=',' read -r -a MODELS <<< "$ONLY"
else
  for d in "$SUB_ROOT"/*/; do
    name=$(basename "$d")
    [[ "$name" == "logs" || "$name" == "zips" ]] && continue
    MODELS+=("$name")
  done
fi
[[ ${#MODELS[@]} -gt 0 ]] || { echo "no model folders found under $SUB_ROOT" >&2; exit 2; }

# Read `name = "..."` from the Solver class in submission.py
solver_name() {
  python3 - "$1" <<'PY'
import re, sys
src = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"class\s+Solver\b.*?^\s+name\s*=\s*['\"]([^'\"]+)['\"]", src, re.S | re.M)
print(m.group(1) if m else "")
PY
}

run() { if [[ $DRY_RUN -eq 1 ]]; then echo "  [dry-run] $*"; else "$@"; fi; }

# ------------------------------------------------------------------ main loop
declare -a SUMMARY=()
ANY_FAIL=0

for model in "${MODELS[@]}"; do
  dir="$SUB_ROOT/$model"
  echo "=== $model ==="
  status="OK"; solver=""; zip_path="-"

  if [[ ! -d "$dir" ]]; then
    echo "  folder not found, skipped"; SUMMARY+=("$model|-|MISSING FOLDER|-"); ANY_FAIL=1; continue
  fi
  if [[ ! -f "$dir/submission.py" ]]; then
    echo "  submission.py not found, skipped"; SUMMARY+=("$model|-|NO submission.py|-"); ANY_FAIL=1; continue
  fi
  if [[ ! -s "$dir/$WEIGHTS" ]]; then
    echo "  $WEIGHTS not found or empty, skipped"; SUMMARY+=("$model|-|NO $WEIGHTS|-"); ANY_FAIL=1; continue
  fi

  solver=$(solver_name "$dir/submission.py")
  if [[ -z "$solver" ]]; then
    echo "  could not read Solver.name from submission.py, skipped"
    SUMMARY+=("$model|?|NO Solver.name|-"); ANY_FAIL=1; continue
  fi
  echo "  Solver.name : $solver"

  # ---- benchopt local test
  if [[ $DO_RUN -eq 1 ]]; then
    solver_file="$SOLVERS_DIR/$model.py"
    log="$LOG_DIR/$model.log"
    echo "  solver copy : $solver_file"
    echo "  log         : $log"
    run cp "$dir/submission.py" "$solver_file"
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "  [dry-run] (cd $COMP_DIR && COMPET_SUBMISSION_DIR=$dir benchopt run tracks/$TRACK -d $DATASET -s $solver)"
    else
      start=$(date +%s)
      if ( cd "$COMP_DIR" && COMPET_SUBMISSION_DIR="$dir" benchopt run "tracks/$TRACK" -d "$DATASET" -s "$solver" ) \
           > "$log" 2>&1; then
        echo "  benchopt    : PASSED ($(( $(date +%s) - start )) s)"
      else
        status="FAILED"; ANY_FAIL=1
        echo "  benchopt    : FAILED ($(( $(date +%s) - start )) s) -- last lines of the log:"
        tail -n 15 "$log" | sed 's/^/      /'
      fi
    fi
    [[ $KEEP_SOLVERS -eq 1 ]] || run rm -f "$solver_file"
  fi

  # ---- flat ZIP for Codabench (only when the test passed, or when not testing)
  if [[ $DO_ZIP -eq 1 && "$status" == "OK" ]]; then
    zip_path="$ZIP_DIR/$model.zip"
    extra=()
    while IFS= read -r f; do extra+=("$f"); done < <(
      find "$dir" -maxdepth 1 -type f ! -name submission.py ! -name "$WEIGHTS" \
           ! -name '*.zip' ! -name '*.log' ! -name '.*' -printf '%p\n' | sort)
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "  [dry-run] zip -j $zip_path submission.py $WEIGHTS ${extra[*]##*/}"
    else
      rm -f "$zip_path"
      if zip -q -j "$zip_path" "$dir/submission.py" "$dir/$WEIGHTS" "${extra[@]}"; then
        echo "  zip         : $zip_path ($(du -h "$zip_path" | cut -f1))"
      else
        status="ZIP FAILED"; ANY_FAIL=1; zip_path="-"; echo "  zip         : FAILED"
      fi
    fi
  fi

  SUMMARY+=("$model|$solver|$status|$zip_path")
  echo
done

# ------------------------------------------------------------------- summary
echo "================================ summary ================================"
printf '%-20s %-18s %-14s %s\n' "folder" "Solver.name" "status" "zip"
printf '%-20s %-18s %-14s %s\n' "------" "-----------" "------" "---"
for row in "${SUMMARY[@]}"; do
  IFS='|' read -r a b c d <<< "$row"
  printf '%-20s %-18s %-14s %s\n' "$a" "$b" "$c" "$d"
done
echo
[[ $DO_RUN -eq 1 && $DRY_RUN -eq 0 ]] && echo "Logs: $LOG_DIR/<folder>.log   benchopt outputs: $TRACK_DIR/outputs/"
[[ $DO_ZIP -eq 1 ]] && echo "Upload the ZIPs in $ZIP_DIR to Codabench (one per model)."
exit $ANY_FAIL
