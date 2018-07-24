LAYERS=269
WIDE_SCALE=4

export CUDA_VISIBLE_DEVICES=0
export MXNET_ENGINE_TYPE=NaiveEngine
export PYTHONPATH=/home/qingsen/incubator-mxnet/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

python benchmark.py --num_layers=${LAYERS} --batch_size 4 --wide_scale=${WIDE_SCALE} --num_loop=6 resnet > log_resnet_${LAYERS}_4_${WIDE_SCALE}
