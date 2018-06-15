LAYERS=18
WIDE_SCALE=8
export MXNET_SWAP_NO_COPY=1
export MXNET_SWAP_THRESHOLD_MULTIPLIER=512

python benchmark.py --num_layers=${LAYERS} --batch_size 4 --wide_scale=${WIDE_SCALE} --num_loop=6 --cold_skip=3 resnet > log_resnet_${LAYERS}_4_${WIDE_SCALE}_no_copy
