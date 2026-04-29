#!/usr/bin/env bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate curobo_env

cd /home/pm/Desktop/Project/robot_stack

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_129_LIB="$CONDA_PREFIX/targets/sbsa-linux/lib"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_129_LIB:$CONDA_PREFIX/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export LD_PRELOAD="$CUDA_129_LIB/libnvrtc.so.12"
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
export PYTHONNOUSERSITE=1

cat > /tmp/curobo_request.json <<'JSON'
{
  "mode": "smoke_test"
}
JSON

python -m isaac_collector.services.curobo_worker \
  --request /tmp/curobo_request.json \
  --output /tmp/curobo_output.json

cat /tmp/curobo_output.json
