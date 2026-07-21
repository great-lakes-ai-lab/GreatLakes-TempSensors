#!/bin/bash
# setup_env.sh — Sets up the deepsensor-gl environment
# Run from the GreatLakes-TempSensors repo root
#
# USAGE:
#   source setup_env.sh
#
# WORKFLOW on Great Lakes:
#   1. Run `source setup_env.sh` on login node first (clones repos, skips env creation)
#   2. Get a compute node: salloc --account=YOUR_ACCOUNT --partition=standard --cpus-per-task=4 --mem=16G --time=01:00:00
#   3. Run `source setup_env.sh` again on compute node (creates env, installs packages)
#
# WORKFLOW on local (Mac):
#   source setup_env.sh  (does everything in one go)

# Ensure this is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]] 2>/dev/null || [[ "$ZSH_EVAL_CONTEXT" == "toplevel" ]] 2>/dev/null; then
    echo "ERROR: This script must be sourced, not executed."
    echo "Usage: source setup_env.sh"
    exit 1
fi

# --- Configuration ---
ENV_NAME="deepsensor-gl"
DEEPSENSOR_REPO="git@github.com:jeremyg19/deepsensor.git"
DEEPSENSOR_BRANCH="fix/local-patches"

# Anchor paths to repo root (where this script lives)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)"
DEEPSENSOR_DIR="$REPO_ROOT/../deepsensor"

# --- Detect conda vs mamba ---
CONDA_CMD=""
if type mamba &>/dev/null; then
    CONDA_CMD="mamba"
elif type conda &>/dev/null; then
    CONDA_CMD="conda"
else
    echo "ERROR: Neither mamba nor conda found."
    return 1
fi
echo "Using: $CONDA_CMD"

# --- Detect environment ---
ON_HPC=false
ON_LOGIN=false
ON_COMPUTE=false

CURR_HOSTNAME=$(hostname)

if [[ "$CURR_HOSTNAME" == *"gl-login"* ]]; then
    ON_HPC=true
    ON_LOGIN=true
elif [[ "$CURR_HOSTNAME" == gl* && "$CURR_HOSTNAME" == *".arc-ts.umich.edu"* ]]; then
    ON_HPC=true
    ON_COMPUTE=true
fi

# --- Functions ---

clone_repos() {
    echo "=== Checking DeepSensor fork ==="
    if [ ! -d "$DEEPSENSOR_DIR" ]; then
        echo "Cloning deepsensor fork..."
        git clone "$DEEPSENSOR_REPO" "$DEEPSENSOR_DIR" || { echo "ERROR: git clone failed"; return 1; }
        cd "$DEEPSENSOR_DIR"
        git checkout "$DEEPSENSOR_BRANCH" || { echo "ERROR: git checkout failed"; cd "$REPO_ROOT"; return 1; }
        cd "$REPO_ROOT"
    else
        echo "DeepSensor repo exists at $DEEPSENSOR_DIR"
        cd "$DEEPSENSOR_DIR"
        git checkout "$DEEPSENSOR_BRANCH"
        git pull origin "$DEEPSENSOR_BRANCH" || echo "WARNING: Could not pull (no network?). Using existing code."
        cd "$REPO_ROOT"
    fi
    echo "DeepSensor fork: OK"
}

create_env() {
    echo "=== Creating conda environment ==="

    if conda env list | grep -q "$ENV_NAME"; then
        echo "Environment '$ENV_NAME' exists, updating..."
        $CONDA_CMD env update -n "$ENV_NAME" -f "$REPO_ROOT/environment-base.yml" --prune
    else
        echo "Creating new environment..."
        $CONDA_CMD env create -f "$REPO_ROOT/environment-base.yml"
    fi

    if ! conda env list | grep -q "$ENV_NAME"; then
        echo "ERROR: Environment creation failed."
        return 1
    fi

    echo "Environment '$ENV_NAME': OK"
}

install_packages() {
    echo "=== Activating environment ==="
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME" || { echo "ERROR: Could not activate $ENV_NAME"; return 1; }

    echo "=== Installing PyTorch ==="
    if [[ $ON_HPC == true ]]; then
        echo "Installing PyTorch with CUDA 12.1..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || { echo "ERROR: PyTorch install failed"; return 1; }
    else
        echo "Installing PyTorch (CPU/MPS)..."
        pip install torch torchvision torchaudio || { echo "ERROR: PyTorch install failed"; return 1; }
    fi

    echo "=== Installing DeepSensor (editable) ==="
    pip install -e "$DEEPSENSOR_DIR"'[torch]' || { echo "ERROR: DeepSensor install failed"; return 1; }

    echo "=== Installing GreatLakes-TempSensors (editable) ==="
    pip install -e "$REPO_ROOT" || { echo "ERROR: GL-TS install failed"; return 1; }

    echo ""
    echo "=== Verifying installation ==="
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    python -c "from deepsensor.data import DataProcessor; print('DeepSensor: OK')"
    python -c "from pipeline.config import load_config; print('GL-TS pipeline: OK')"
}

# --- Main logic ---

if [[ $ON_LOGIN == true ]]; then
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Detected: Great Lakes LOGIN node                           ║"
    echo "║  Will clone repos only (env creation needs compute node)    ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    clone_repos || return 1

    # Check if env already exists
    if conda env list | grep -q "$ENV_NAME"; then
        echo ""
        echo "Environment '$ENV_NAME' already exists."
        echo "To activate: conda activate $ENV_NAME"
    else
        echo ""
        echo "══════════════════════════════════════════════════════════════"
        echo "  Next steps:"
        echo "    1. Get a compute node:"
        echo "       salloc --account=YOUR_ACCOUNT --partition=standard \\"
        echo "              --cpus-per-task=4 --mem=16G --time=01:00:00"
        echo ""
        echo "    2. On the compute node, run again:"
        echo "       cd $REPO_ROOT"
        echo "       source setup_env.sh"
        echo "══════════════════════════════════════════════════════════════"
    fi

elif [[ $ON_COMPUTE == true ]]; then
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Detected: Great Lakes COMPUTE node                         ║"
    echo "║  Will create environment and install packages               ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    # Verify repos are cloned
    if [ ! -d "$DEEPSENSOR_DIR" ]; then
        echo "ERROR: DeepSensor repo not found at $DEEPSENSOR_DIR"
        echo "Please run 'source setup_env.sh' on a login node first to clone repos."
        return 1
    fi

    create_env || return 1
    install_packages || return 1

else
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Detected: Local machine                                    ║"
    echo "║  Will do full setup (clone + env + install)                 ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    clone_repos || return 1
    create_env || return 1
    install_packages || return 1
fi

echo ""
echo "=== Done ==="