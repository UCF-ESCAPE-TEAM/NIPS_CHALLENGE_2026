#!/usr/bin/env bash
# fix_emg2pose_sidecars.sh
#
# Repair an incomplete emg2pose (NEMAR nm000281) download used by NeuralBench.
#
# The EEG-Dash downloader sometimes finishes with a few sidecar files missing
# (e.g. *_channels.tsv, *_events.tsv) and then refuses to refetch them because
# a "success" marker says the dataset is complete. NeuralBench then fails with
#   "picks ('misc') could not be interpreted ..."
# This script compares the local BIDS tree against the dataset's public
# repository (github.com/nemarDatasets/nm000281), downloads every missing or
# empty text file, and reports anything it cannot fix (the large .bdf signal
# files are not on GitHub).
#
# Usage:
#   bash fix_emg2pose_sidecars.sh [BIDS_ROOT] [--dry-run] [--jobs N] [--with-json]
#
#   BIDS_ROOT   folder containing sub-*/ (default: taken from
#               ~/.neuralbench/config.json -> DATA_DIR/Salter2024Emg2pose/download/nm000281)
#   --dry-run   only report, download nothing
#   --jobs N    parallel downloads (default 8)
#   --with-json also fetch the *_emg.json sidecars (not needed by NeuralBench)
#
# Requirements: bash, git, curl, python3 (only to read the NeuralBench config).
# Safe to rerun: existing non-empty files are never touched.

set -euo pipefail

REPO_URL="https://github.com/nemarDatasets/nm000281.git"
RAW_BASE="https://raw.githubusercontent.com/nemarDatasets/nm000281"
REF="${NM000281_REF:-main}"      # override with NM000281_REF=<branch|tag|commit>

ROOT=""
DRY_RUN=0
JOBS=8
WITH_JSON=0

for arg in "$@"; do
  case "$arg" in
    --dry-run)   DRY_RUN=1 ;;
    --with-json) WITH_JSON=1 ;;
    --jobs=*)    JOBS="${arg#--jobs=}" ;;
    --jobs)      ;;   # handled below
    -h|--help)   sed -n '2,24p' "$0"; exit 0 ;;
    *)
      if [[ "${prev:-}" == "--jobs" ]]; then JOBS="$arg"
      elif [[ -z "$ROOT" ]]; then ROOT="$arg"
      else echo "unexpected argument: $arg" >&2; exit 2; fi ;;
  esac
  prev="$arg"
done

# ---------------------------------------------------------------- locate data
if [[ -z "$ROOT" ]]; then
  CFG="${NEURALBENCH_CONFIG:-$HOME/.neuralbench/config.json}"
  if [[ ! -f "$CFG" ]]; then
    echo "No BIDS_ROOT given and $CFG not found." >&2; exit 2
  fi
  DATA_DIR=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['DATA_DIR'])" "$CFG")
  ROOT="$DATA_DIR/Salter2024Emg2pose/download/nm000281"
fi
ROOT="${ROOT%/}"
if [[ ! -d "$ROOT" ]] || ! ls -d "$ROOT"/sub-* >/dev/null 2>&1; then
  echo "No BIDS tree (sub-*/) found under: $ROOT" >&2; exit 2
fi
echo "BIDS root : $ROOT"
echo "Reference : $REPO_URL @ $REF"

for tool in git curl; do
  command -v "$tool" >/dev/null || { echo "$tool is required" >&2; exit 2; }
done

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# ------------------------------------------------- authoritative file listing
echo "Fetching the file list from the dataset repository (no data is cloned)..."
git clone -q --filter=blob:none --no-checkout --depth 1 --branch "$REF" "$REPO_URL" "$WORK/repo"
git -C "$WORK/repo" ls-tree -r --name-only HEAD > "$WORK/all_files.txt"
COMMIT=$(git -C "$WORK/repo" rev-parse --short HEAD)
echo "Repository commit: $COMMIT  ($(wc -l < "$WORK/all_files.txt") files listed)"

# Text files NeuralBench's loader reads, for the recordings present locally:
#   per recording : *_channels.tsv, *_events.tsv  (+ *_emg.json with --with-json)
#   per session   : sub-XX_ses-YY_scans.tsv
#   dataset level : participants.tsv, participants.json, dataset_description.json
# Recordings whose .bdf is absent are reported separately (see summary).
grep -E '_emg\.bdf$' "$WORK/all_files.txt" > "$WORK/bdf.txt" || true
SUFFIXES="channels.tsv events.tsv"
[[ $WITH_JSON -eq 1 ]] && SUFFIXES="$SUFFIXES emg.json"
{
  printf '%s\n' participants.tsv participants.json dataset_description.json
  find "$ROOT" -name "*_emg.bdf" -size +0 | while IFS= read -r f; do
    rel="${f#$ROOT/}"; stem="${rel%_emg.bdf}"
    for sfx in $SUFFIXES; do echo "${stem}_$sfx"; done
    ses_dir=$(dirname "$(dirname "$rel")")                 # sub-XX/ses-YY
    sub=$(basename "$(dirname "$ses_dir")"); ses=$(basename "$ses_dir")
    echo "$ses_dir/${sub}_${ses}_scans.tsv"
  done
} | sort -u > "$WORK/candidates.txt"
# keep only paths the repository actually has (guards against odd local names)
grep -F -x -f "$WORK/candidates.txt" "$WORK/all_files.txt" > "$WORK/wanted.txt" || true

# ----------------------------------------------------- find what is missing
: > "$WORK/to_fetch.txt"
while IFS= read -r rel; do
  f="$ROOT/$rel"
  if [[ ! -s "$f" ]]; then      # missing or empty
    echo "$rel" >> "$WORK/to_fetch.txt"
  fi
done < "$WORK/wanted.txt"

: > "$WORK/missing_bdf.txt"
while IFS= read -r rel; do
  [[ -s "$ROOT/$rel" ]] || echo "$rel" >> "$WORK/missing_bdf.txt"
done < "$WORK/bdf.txt"

N_TEXT=$(wc -l < "$WORK/to_fetch.txt")
N_BDF=$(wc -l < "$WORK/missing_bdf.txt")
N_BDF_TOTAL=$(wc -l < "$WORK/bdf.txt")
echo
echo "Signal files (.bdf) present : $((N_BDF_TOTAL - N_BDF)) / $N_BDF_TOTAL"
echo "Text files missing or empty : $N_TEXT"

if [[ $N_TEXT -gt 0 ]]; then
  echo "----- missing text files -----"
  cat "$WORK/to_fetch.txt"
  echo "------------------------------"
fi

# ----------------------------------------------------------------- download
FAILED_LIST="$ROOT/../fix_sidecars_failed.txt"
if [[ $N_TEXT -gt 0 && $DRY_RUN -eq 0 ]]; then
  echo "Downloading $N_TEXT file(s) with $JOBS parallel jobs..."
  export ROOT RAW_BASE REF
  : > "$FAILED_LIST"
  fetch_one() {
    rel="$1"; dest="$ROOT/$rel"; tmp="$dest.part"
    mkdir -p "$(dirname "$dest")"
    if curl -fsSL --retry 3 --retry-delay 2 -o "$tmp" "$RAW_BASE/$REF/$rel" && [[ -s "$tmp" ]]; then
      mv -f "$tmp" "$dest"; echo "ok      $rel"
    else
      rm -f "$tmp"; echo "FAILED  $rel"; echo "$rel" >> "$FAILED_LIST"
    fi
  }
  export -f fetch_one
  export FAILED_LIST
  xargs -a "$WORK/to_fetch.txt" -P "$JOBS" -I{} bash -c 'fetch_one "$@"' _ {}
  N_FAILED=$(wc -l < "$FAILED_LIST")
  if [[ $N_FAILED -eq 0 ]]; then rm -f "$FAILED_LIST"; fi
else
  N_FAILED=0
fi

# ------------------------------------------------------------------ summary
echo
echo "================= summary ================="
echo "Text files fetched : $(( DRY_RUN ? 0 : N_TEXT - N_FAILED ))"
echo "Text files failed  : $N_FAILED$( [[ $N_FAILED -gt 0 ]] && echo "   (list: $FAILED_LIST)" )"
echo "Signal files absent: $N_BDF"
if [[ $N_BDF -gt 0 ]]; then
  cp "$WORK/missing_bdf.txt" "$ROOT/../missing_bdf.txt"
  echo "  The .bdf files are not on GitHub. Their list is in $ROOT/../missing_bdf.txt"
  echo "  To refetch them, delete the downloader's success marker and rerun the download:"
  echo "    rm \"$ROOT/../eegdash_nm000281_success_download.txt\""
  echo "    neuralbench emg pose -m <model> --download"
fi
if [[ $DRY_RUN -eq 1 ]]; then echo "(dry run: nothing was downloaded)"; fi
if [[ $N_TEXT -eq 0 && $N_BDF -eq 0 ]]; then echo "Dataset is complete."; fi
