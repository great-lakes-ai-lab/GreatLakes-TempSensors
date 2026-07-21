#!/bin/bash
# setup_env.sh — Sets up the deepsensor-gl environment
# Run from the GreatLakes-TempSensors repo root

# WORKFLOW on Great Lakes:
#   1. Run `source setup_env.sh` on login node first (clones repos, skips env creation)
#   2. Get a compute node: salloc --account=YOUR_ACCOUNT --partition=standard --cpus-per-task=4 --mem=16G --time=01:00:00
#   3. Run `source setup_env.sh` again on compute node (creates env, installs packages)
#
# WORKFLOW on local (Mac):
#   source setup_env.sh  (does everything in one go)

# Prevent set -e from killing the shell when sourced
# set -e  # REMOVED

# Ensure this is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced, not executed."
    echo "Usage: source setup_env.sh"
    exit 1
fi

ENV_NAME="deepsensor-gl"
DEEPSENSOR_REPO="git@github.com:jeremyg19/deepsensor.git"
DEEPSENSOR_BRANCH="fix/local-patches"

# Anchor paths to repo root (where this script lives)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPSENSOR_DIR="$REPO_ROOT/../deepsensor"

# --- Detect environment ---
ON_HPC=false
ON_LOGIN=false
ON_COMPUTE=false

HOSTNAME=$(hostname)

if [[ "$HOSTNAME" == *"gl-login"* ]]; then
    ON_HPC=true
    ON_LOGIN=true
    echo "ON HPC LOGIN NODE"
elif [[ "$HOSTNAME" == gl* && "$HOSTNAME" == *".arc-ts.umich.edu"* ]]; then
    ON_HPC=true
    ON_COMPUTE=true
    echo "ON HPC COMPUTE NODE"
else
    echo "ON LOCAL MACHINE"
fi

# --- Step 1: Clone repos (needs network — login node or local) ---
clone_repos() {
    echo "=== Checking DeepSensor fork ==="
    if [ ! -d "$DEEPSENSOR_DIR" ]; then
        echo "Cloning deepsensor fork..."
        git clone $DEEPSENSOR_REPO $DEEPSENSOR_DIR
        cd $DEEPSENSOR_DIR
        git checkout $DEEPSENSOR_BRANCH
        cd -
    else
        echo "DeepSensor repo exists at $DEEPSENSOR_DIR"
        cd $DEEPSENSOR_DIR
        git checkout $DEEPSENSOR_BRANCH
        git pull origin $DEEPSENSOR_BRANCH || echo "WARNING: Could not pull (no network?). Using existing code."
        cd -
    fi
}

# --- Step 2: Create conda environment (needs resources — compute node or local) ---
create_env() {
    echo "=== Creating conda environment ==="
    echo "DEBUG: CONDA_CMD='$CONDA_CMD'"
    echo "DEBUG: running '$CONDA_CMD env create -f environment-base.yml'"
    # Check if environment already exists
    if conda env list | grep -q "^$ENV_NAME "; then
        echo "Environment '$ENV_NAME' exists, updating..."
        $CONDA_CMD env update -n $ENV_NAME -f environment-base.yml --prune
    else
        echo "Creating new environment..."
        $CONDA_CMD env create -f environment-base.yml
    fi

    # HPC-only: add CUDA support
    if [[ $ON_HPC == true ]]; then
        echo "=== Adding CUDA support ==="
        $CONDA_CMD env update -n $ENV_NAME -f environment-hpc.yml
    fi
}


# --- Step 3: Activate and pip install editable packages ---
install_packages() {
    echo "=== Activating environment ==="
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME

    echo "=== Installing DeepSensor (editable) ==="
    pip install -e "$DEEPSENSOR_DIR[torch]"

    echo "=== Installing GreatLakes-TempSensors (editable) ==="
    pip install -e .

    echo ""
    echo "=== Setup complete ==="
    python -c "from deepsensor.data import DataProcessor; print('DeepSensor: OK')"
    python -c "from utils.coordinates import standardize_coords; print('GL-TS utils: OK')"
    python -c "from pipeline.config import load_config; print('GL-TS pipeline: OK')"
}

# --- Main logic ---
if [[ $ON_LOGIN == true ]]; then
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Detected: Great Lakes LOGIN node                           ║"
    echo "║  Will clone repos only (env creation needs compute node)    ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    clone_repos

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
        echo "       cd $(pwd)"
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

    create_env
    install_packages

else
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Detected: Local machine                                    ║"
    echo "║  Will do full setup (clone + env + install)                 ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    clone_repos
    create_env
    install_packages
fi
