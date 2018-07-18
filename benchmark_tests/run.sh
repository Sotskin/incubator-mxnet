# Suggested Setting: Layers=269 Batch_size=4 Wide_scale=4 
LAYERS=$1
BATCH_SIZE=$2
WIDE_SCALE=$3

echo "LAYERS = ${LAYERS}"
echo "BATCH_SIZE = ${BATCH_SIZE}"
echo "WIDE_SCALE = ${WIDE_SCALE}"

export SWAP_ALGORITHM=NaiveHistory
export MXNET_ENGINE_TYPE=NaiveEngine
export PYTHONPATH=/home/sotskin/incubator-mxnet/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export CUDA_VISIBLE_DEVICES=1

python benchmark.py --num_gpus=1 --num_layers=${LAYERS} --batch_size=${BATCH_SIZE} --wide_scale=${WIDE_SCALE} --num_loop=6 resnet > log_resnet_${LAYERS}_4_${WIDE_SCALE}
