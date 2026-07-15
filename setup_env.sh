#!/bin/bash
# setup_env.sh — Sets up the deepsensor-gl environment
# Run from the GreatLakes-TempSensors repo root

set -e

ENV_NAME="deepsensor-gl"
DEEPSENSOR_REPO="git@github.com:jeremyg19/deepsensor.git"
DEEPSENSOR_BRANCH="fix/local-patches"
DEEPSENSOR_DIR="../deepsensor"  # relative to this repo

echo "=== Creating conda environment ==="
mamba env create -f environment-base.yml || echo "Environment already exists, updating..."
mamba env update -f environment-base.yml --prune

# HPC-only: add CUDA support
if command -v nvidia-smi &> /dev/null || [[ $(hostname) == *"greatlakes"* ]] || [[ $(hostname) == *"gl-"* ]]; then
    echo "=== Detected HPC, adding CUDA support ==="
    mamba env update -f environment-hpc.yml
fi

echo "=== Activating environment ==="
eval "$(conda shell.bash hook)"
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
echo "Activate with: mamba activate $ENV_NAME"
python -c "from deepsensor.data import DataProcessor; print('DeepSensor: OK')"
python -c "from utils.coordinates import standardize_coords; print('GL-TS utils: OK')"
python -c "from pipeline.config import load_config; print('GL-TS pipeline: OK')"