# Suggested Setting: ./lstm.sh 32 30 2 1024
BATCH_SIZE=$1
SEQ_LEN=$2
LAYERS=$3
HIDDEN_SIZE=$4
INPUT_SIZE=512
CLASSES=512
NUM_GPU=1
NUM_LOOPS=30
COLD_SKIP=5
USE_MOMENTUM=0

echo "BATCH_SIZE = ${BATCH_SIZE}"
echo "SEQUENCE_LENGTH = ${SEQ_LEN}"
echo "LAYERS = ${LAYERS}"
echo "HIDDEN_SIZE = ${HIDDEN_SIZE}"

export SWAP_ALGORITHM=NaiveHistory
export MXNET_ENGINE_TYPE=NaiveEngine
export PYTHONPATH=/home/karl/incubator-mxnet/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export CUDA_VISIBLE_DEVICES=0

python tofu_lstm_benchmark.py --cold_skip=${COLD_SKIP} --input_size=${INPUT_SIZE} --num_classes=${CLASSES} --use_momentum=${USE_MOMENTUM} --num_gpus=${NUM_GPU} --hidden_size=${HIDDEN_SIZE} --num_layers=${LAYERS} --batch_size=${BATCH_SIZE} --seq_len=${SEQ_LEN} --num_loop=${NUM_LOOPS} > log_lstm_${LAYERS}_${BATCH_SIZE}_${SEQ_LEN}_${HIDDEN_SIZE}
