#!/usr/bin/env python3
"""
orchestrator.py

Run the full pipeline Stage1 -> Stage2 -> Stage3 -> Stage4 in order.

Assumptions:
 - All stage scripts live in the same directory and are named:
     stage1: indeed_full_scraper_bs4_fixed.py       (or your chosen stage1 filename)
     stage2: Stage2DataCleaningAndResumeParsing.py  (or Stage2DataCleaningAndResumeParsing_NoNLTK.py)
     stage3: Stage3_embed_openai.py
     stage4: Stage4_similarity.py
 - Required environment variables (RAPIDAPI_KEY, OPENAI_API_KEY) are set externally, OR
   you can edit the script to set them (not recommended for secrets).
 - You run this from the repo directory containing the scripts.

Usage:
  python orchestrator.py
  python orchestrator.py --resume myresume.pdf
  python orchestrator.py --skip-stage1 --skip-stage3
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import shutil
import time

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
# Filenames for stage scripts (must be in same folder)
STAGE1_SCRIPT = "Stage1DataAcquisition.py"       # Stage 1: fetch API data
STAGE2_SCRIPT = "Stage2DataCleaningAndResumeParsing.py"  # Stage 2: cleaning (use whichever variant you prefer)
STAGE3_SCRIPT = "Stage3OpenAIEmbedding.py"                 # Stage 3: embeddings (OpenAI)
STAGE4_SCRIPT = "Stage4Similarity.py"                   # Stage 4: similarity / top jobs

# Files produced by stages (or expected) - used for quick checks
STAGE1_RAW = Path("indeed_response_raw.json")            # Stage1 raw
STAGE1_OUT = Path("stage1_jobs.json")                    # Stage1 flattened output (if produced)
STAGE2_OUT_JSON = Path("cleaned_jobs.json")              # Stage2 cleaned jobs
STAGE2_OUT_CSV  = Path("cleaned_jobs.csv")
STAGE2_RESUME_PROCESSED = Path("processed_resume.txt")   # Stage2 resume processed text
STAGE3_JOBS_EMBED = Path("jobs_embeddings.jsonl")        # Stage3 embeddings
STAGE3_RESUME_EMBED = Path("resume_embedding.json")      # Stage3 resume embedding
STAGE4_TOP = Path("top_jobs.json")                       # Stage4 top results

# Pipeline behavior
DEFAULT_RESUME_PATH = "Cole_Gruenke_Resume.pdf"   # the resume file to pass into Stage 2 (if present)
STOP_ON_ERROR = True                 # if True, stop the pipeline on first stage failure
SKIP_IF_OUTPUT_EXISTS = True         # if True, skip a stage if its expected output already exists

# Limits to avoid over-using APIs
MAX_STAGE1_RUNS = 1  # do not run stage1 more than this many times in a single orchestrator run (safety)
# ---------------------------------------------------------------------

def which_python():
    """Return the python executable used to run this orchestrator."""
    return sys.executable

def run_script(script: str, args: list = None, env: dict = None, timeout: int = None) -> int:
    """
    Run a script as a subprocess. Returns the exit code.
    Prints stdout/stderr live (streamed).
    """
    args = args or []
    cmd = [which_python(), script] + args
    print(f"\n=== Running: {' '.join(cmd)} ===")
    try:
        # stream output to the terminal
        proc = subprocess.run(cmd, env=env, check=False, timeout=timeout)
        print(f"=== Exit code: {proc.returncode} ===\n")
        return proc.returncode
    except subprocess.TimeoutExpired:
        print(f"ERROR: Script {script} timed out after {timeout} seconds.")
        return 124
    except Exception as e:
        print(f"ERROR: Running script {script} failed: {e}")
        return 1

def assert_exists_warn(path: Path, name: str):
    if not path.exists():
        print(f"WARNING: expected {name} ({path}) not found.")

def main_cli():
    parser = argparse.ArgumentParser(description="Orchestrate Stage1->Stage4 pipeline.")
    parser.add_argument("--resume", type=str, default=DEFAULT_RESUME_PATH, help="Path to resume file (pdf or txt).")
    parser.add_argument("--skip-stage1", action="store_true", help="Skip Stage 1 (fetch).")
    parser.add_argument("--skip-stage2", action="store_true", help="Skip Stage 2 (clean).")
    parser.add_argument("--skip-stage3", action="store_true", help="Skip Stage 3 (embed).")
    parser.add_argument("--skip-stage4", action="store_true", help="Skip Stage 4 (similarity).")
    parser.add_argument("--continue-on-error", action="store_true", help="Do not stop pipeline on stage error.")
    parser.add_argument("--force-stage1", action="store_true", help="Force run Stage1 even if outputs exist (overrides skip logic).")
    args = parser.parse_args()
    return args

def orchestrate(args):
    cwd = Path.cwd()
    python_exec = which_python()
    # basic checks
    print("Orchestrator running in:", cwd)
    print("Using Python:", python_exec)
    print("Stage scripts (must be in same directory):")
    print("  ", STAGE1_SCRIPT)
    print("  ", STAGE2_SCRIPT)
    print("  ", STAGE3_SCRIPT)
    print("  ", STAGE4_SCRIPT)
    print()

    resume_path = Path(args.resume) if args.resume else Path(DEFAULT_RESUME_PATH)
    if resume_path.exists():
        print("Found resume:", resume_path)
    else:
        print("Resume not found at", resume_path, "- Stage 2 resume parsing will be skipped unless you provide it.")
    print()

    continue_on_error = args.continue_on_error
    stop_on_error = not continue_on_error

    # track counts (safety)
    stage1_runs = 0

    # Stage 1
    if args.skip_stage1:
        print("Skipping Stage 1 (per CLI).")
    else:
        # skip logic if outputs exist
        if SKIP_IF_OUTPUT_EXISTS and STAGE1_OUT.exists() and not args.force_stage1:
            print(f"Skipping Stage 1 because {STAGE1_OUT} exists. Use --force-stage1 to rerun.")
        else:
            if stage1_runs >= MAX_STAGE1_RUNS:
                print(f"Refusing to run Stage 1 more than {MAX_STAGE1_RUNS} times in one orchestrator run.")
            else:
                # ensure RAPIDAPI_KEY present
                if not os.getenv("RAPIDAPI_KEY"):
                    print("WARNING: RAPIDAPI_KEY not set in environment. Stage 1 may fail.")
                rc = run_script(STAGE1_SCRIPT)
                stage1_runs += 1
                if rc != 0 and stop_on_error:
                    print("Stage 1 failed. Stopping pipeline.")
                    return rc

    # Stage 2
    if args.skip_stage2:
        print("Skipping Stage 2 (per CLI).")
    else:
        if SKIP_IF_OUTPUT_EXISTS and STAGE2_OUT_JSON.exists():
            print(f"Skipping Stage 2 because {STAGE2_OUT_JSON} exists.")
        else:
            # pass resume path to Stage 2 via CLI argument if resume exists
            stage2_args = []
            if resume_path.exists():
                stage2_args = [str(resume_path)]
            rc = run_script(STAGE2_SCRIPT, args=stage2_args)
            if rc != 0 and stop_on_error:
                print("Stage 2 failed. Stopping pipeline.")
                return rc

    # Stage 3 (embeddings)
    if args.skip_stage3:
        print("Skipping Stage 3 (per CLI).")
    else:
        if SKIP_IF_OUTPUT_EXISTS and STAGE3_JOBS_EMBED.exists() and STAGE3_RESUME_EMBED.exists():
            print(f"Skipping Stage 3 because {STAGE3_JOBS_EMBED} and {STAGE3_RESUME_EMBED} exist.")
        else:
            # require OpenAI key for Stage 3
            if not os.getenv("OPENAI_API_KEY"):
                print("WARNING: OPENAI_API_KEY not set in environment. Skipping Stage 3 to avoid accidental calls.")
                if stop_on_error:
                    print("Stop_on_error enabled and OPENAI_API_KEY missing. Exiting.")
                    return 1
            else:
                rc = run_script(STAGE3_SCRIPT)
                if rc != 0 and stop_on_error:
                    print("Stage 3 failed. Stopping pipeline.")
                    return rc

    # Stage 4 (similarity)
    if args.skip_stage4:
        print("Skipping Stage 4 (per CLI).")
    else:
        # Stage 4 only needs the embeddings files
        if SKIP_IF_OUTPUT_EXISTS and STAGE4_TOP.exists():
            print(f"Skipping Stage 4 because {STAGE4_TOP} exists.")
        else:
            # pass top N maybe later; for now just run
            rc = run_script(STAGE4_SCRIPT)
            if rc != 0 and stop_on_error:
                print("Stage 4 failed. Stopping pipeline.")
                return rc

    print("\nPipeline completed.")
    # print where outputs are
    print("Outputs:")
    for p in [STAGE1_RAW, STAGE1_OUT, STAGE2_OUT_JSON, STAGE2_OUT_CSV, STAGE2_RESUME_PROCESSED, STAGE3_JOBS_EMBED, STAGE3_RESUME_EMBED, STAGE4_TOP]:
        if p.exists():
            print("  ✓", p)
        else:
            print("  -", p, "(not present)")
    return 0

if __name__ == "__main__":
    args = main_cli()
    rc = orchestrate(args)
    sys.exit(rc)
