LAYERS=18
WIDE_SCALE=8

python benchmark.py --num_layers=${LAYERS} --batch_size 4 --wide_scale=${WIDE_SCALE} --num_loop=6 resnet > log_resnet_${LAYERS}_4_${WIDE_SCALE}
