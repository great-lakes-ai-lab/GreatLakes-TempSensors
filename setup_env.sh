
#!/bin/bash
# setup_env.sh — Sets up the deepsensor-gl environment
# Run from the GreatLakes-TempSensors repo root
#
# USAGE:
#   source setup_env.sh
#
# NOTE: Must be sourced (not executed) for conda activate to work

set -e

ENV_NAME="deepsensor-gl"
DEEPSENSOR_REPO="git@github.com:jeremyg19/deepsensor.git"
DEEPSENSOR_BRANCH="fix/local-patches"
DEEPSENSOR_DIR="../deepsensor"

# Detect conda vs mamba
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
else
    echo "mamba not found, using conda"
    CONDA_CMD="conda"
fi

echo "=== Creating conda environment ==="
$CONDA_CMD env create -f environment-base.yml 2>/dev/null || $CONDA_CMD env update -f environment-base.yml --prune

# HPC-only: add CUDA support
if command -v nvidia-smi &> /dev/null || [[ $(hostname) == *"greatlakes"* ]] || [[ $(hostname) == *"gl-"* ]]; then
    echo "=== Detected HPC, adding CUDA support ==="
    $CONDA_CMD env update -f environment-hpc.yml
fi

echo "=== Activating environment ==="
conda activate $ENV_NAME

echo "=== Installing DeepSensor (editable) ==="
if [ ! -d "$DEEPSENSOR_DIR" ]; then
    echo "Cloning deepsensor fork..."
    git clone $DEEPSENSOR_REPO $DEEPSENSOR_DIR
    cd $DEEPSENSOR_DIR
    git checkout $DEEPSENSOR_BRANCH
    cd -
else
    echo "DeepSensor repo exists, pulling latest..."
    cd $DEEPSENSOR_DIR
    git checkout $DEEPSENSOR_BRANCH
    git pull origin $DEEPSENSOR_BRANCH
    cd -
fi
pip install -e "$DEEPSENSOR_DIR[torch]"

echo "=== Installing GreatLakes-TempSensors (editable) ==="
pip install -e .

echo ""
echo "=== Setup complete ==="
python -c "from deepsensor.data import DataProcessor; print('DeepSensor: OK')"
python -c "from utils.coordinates import standardize_coords; print('GL-TS utils: OK')"
python -c "from pipeline.config import load_config; print('GL-TS pipeline: OK')"