export I_MPI_FABRICS=shm:dapl
export I_MPI_DAPL_PROVIDER=ofa-v2-ib0
export I_MPI_DYNAMIC_CONNECTION=0
export LD_LIBRARY_PATH=/opt/intel/lib/intel64:$LD_LIBRARY_PATH
export PYTHONPATH=/home/tofu/mxnet/python
export NNVM_EXEC_MATCH_RANGE=0
export OMP_NUM_THREADS=8
#export I_MPI_PLATFORM=snb
#export I_MPI_RDMA_RNDV_WRITE=0
#export MXNET_CPU_PRIORITY_NTHREADS=8
#export RDMA_IBA_EAGER_THRESHOLD=8192
#export I_MPI_EAGER_THRESHOLD=8192
