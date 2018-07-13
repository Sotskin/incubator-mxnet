LAYERS=269
WIDE_SCALE=4

export PYTHONPATH=/home/karl/debug_naive/incubator-mxnet/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

python benchmark.py --num_layers=${LAYERS} --batch_size 4 --wide_scale=${WIDE_SCALE} --num_loop=6 resnet > log_resnet_${LAYERS}_4_${WIDE_SCALE}
