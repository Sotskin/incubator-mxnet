# Suggested Setting: Layers=269 Batch_size=4 Wide_scale=4 
LAYERS=$1
BATCH_SIZE=$2
WIDE_SCALE=$3
SWAP="NaiveHistory"
PREFETCH="ComputePrefetch"
STEPS=30

echo "LAYERS = ${LAYERS}"
echo "BATCH_SIZE = ${BATCH_SIZE}"
echo "WIDE_SCALE = ${WIDE_SCALE}"

export PREFETCH_STEP_AHEAD=${STEPS}
export PREFETCH_ALGORITHM=${PREFETCH}
export SWAP_ALGORITHM=${SWAP}
export MXNET_ENGINE_TYPE=NaiveEngine
export PYTHONPATH=/home/sotskin/incubator-mxnet/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export CUDA_VISIBLE_DEVICES=1

python benchmark.py --num_gpus=1 --num_layers=${LAYERS} --batch_size=${BATCH_SIZE} --wide_scale=${WIDE_SCALE} --num_loop=10 resnet > log_resnet_${LAYERS}_${BATCH_SIZE}_${WIDE_SCALE}_${SWAP}_${PREFETCH}_${STEPS}

