from __future__ import print_function
import argparse
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Process
import mxnet as mx
import numpy as np
import time

BATCH_SIZE = 1024
WEIGHT_SIZE = 1024
NUM_LAYERS = 5
NUM_ITERATIONS = 2
NUM_IGNORED_ITERATIONS = 1
EXPERIMENT_NAME = ""

class Experiment:
    def __init__(self, iterations, ignored_iterations, name):
        self.iterations = iterations
        self.ignored_iterations = ignored_iterations
        self.count = 0
        self.exps = []
        self.name = name

    def __enter__(self):
        self.count += 1
        if (self.count > self.ignored_iterations):
            self.begin = time.time()

    def __exit__(self, type, value, traceback):
        if (self.count > self.ignored_iterations):
            end = time.time()
            self.exps.append(end - self.begin)

    def Histogram(self):
        min = int(math.floor(self.exps[-1] * 10))
        max = int(math.ceil(self.exps[0] * 10))
        interval = (max - min) / 10.0
        hist = []
        for t in self.exps:
            if interval > 0:
                idx = int((t * 10 - min) / interval)
            else:
                idx = 0
            hist.append((min + idx * interval) / 10.0)
        plt.hist(hist)
        plt.title(self.name)
        plt.xlabel("Seconds")
        plt.ylabel("Frequency")
        plt.savefig("%s.png" % self.name)

    def Summary(self):
        assert len(self.exps) == (self.iterations - self.ignored_iterations), len(self.exps)
        self.exps.sort()
        self.Histogram()
        exps = np.asarray(self.exps)
        avg = exps.mean()
        std = exps.std()
        line = 'avg: %f, std: %f\n' % (avg, std)
        print(line)
        with open('%s.txt' % self.name, 'w') as fp:
            fp.write(line)


def MLP_MP(addresses, worker_index):
    n_workers = len(addresses)
    group2ctx = {'group:%d' % i : mx.cpu(0, addresses[i])
                 for i in range(n_workers)}
    arg_arrays = {}
    arg_grads = {}
    grad_reqs = {}
    data = []
    for i in range(n_workers):
        with mx.AttrScope(ctx_group='group:%d' % i):
            data.append(mx.symbol.ones((BATCH_SIZE, WEIGHT_SIZE / n_workers),
                                       dtype=np.float32))

    weight_shape = (WEIGHT_SIZE, WEIGHT_SIZE / n_workers)
    activations = [None for k in range(n_workers)]
    for l in range(NUM_LAYERS):
        for w in range(n_workers):
            with mx.AttrScope(ctx_group='group:%d' % w):
                var_name = 'w_%d_%d' % (l, w)
                weight = mx.symbol.Variable(var_name, shape=weight_shape,
                                            dtype=np.float32)
                # activations[w] = mx.symbol.dot(data[w], weight)
                activations[w] = mx.symbol.FullyConnected(
                                     data=data[w], weight=weight,
                                     num_hidden=WEIGHT_SIZE, no_bias=True)
                activations[w] = mx.symbol.SliceChannel(
                                     activations[w], axis=1,
                                     num_outputs=n_workers)
                arg_arrays[var_name] = mx.nd.ones(weight_shape, dtype=np.float32)
                arg_grads[var_name] = mx.nd.empty(weight_shape, dtype=np.float32)
                grad_reqs[var_name] = 'write'

        for w in range(n_workers):
            with mx.AttrScope(ctx_group='group:%d' % w):
                all_parts = [activations[i][w] for i in range(n_workers)]
                # all_parts2 = [activations[i][w] + 1 for i in range(n_workers)]
                # data[w] = sum(all_parts) + sum(all_parts2)
                data[w] = mx.symbol.ElementWiseSum(*all_parts)

    net = mx.symbol.Group(data)
    arg_shapes, out_shapes, aux_shapes = net.infer_shape()
    arg_types, out_types, aux_types = net.infer_type()
    executor = net.bind(ctx=mx.cpu(0, addresses[worker_index]), args=arg_arrays,
                        #args_grad=arg_grads, grad_req=grad_reqs,
                        group2ctx=group2ctx)
    exp = Experiment(NUM_ITERATIONS, NUM_IGNORED_ITERATIONS, EXPERIMENT_NAME)
    for i in range(NUM_ITERATIONS):
        with exp:
            output = executor.forward()
            output[0].wait_to_read()
        print("=" * 30)
        print("Finish an iteration %d" % i)
    exp.Summary()

def MLP_DP(addresses, worker_index):
    pass


def Single():
    data0 = mx.symbol.ones((BATCH_SIZE, WEIGHT_SIZE), dtype=np.float32)
    weight_shape = (WEIGHT_SIZE, WEIGHT_SIZE)
    arg_arrays = {}
    for i in range(NUM_LAYERS):
        var_name = 'w_%d' % i
        weight = mx.symbol.Variable(var_name, shape=weight_shape,
                                    dtype=np.float32)
        # activation = mx.symbol.dot(data0, weight)
        activation = mx.symbol.FullyConnected(data=data0, weight=weight,
                                              num_hidden=WEIGHT_SIZE,
                                              no_bias=True)
        #activations = mx.symbol.SliceChannel(activation, axis=0, num_outputs=2)
        #activation = mx.symbol.ElementWiseSum(*activations)
        #activation = activations[0] + activations[1]
        arg_arrays[var_name] = mx.nd.ones(weight_shape, dtype=np.float32)
        data0 = activation

    # net = mx.symbol.Group(data0)
    net = data0
    arg_shapes, out_shapes, aux_shapes = net.infer_shape()
    arg_types, out_types, aux_types = net.infer_type()
    executor = net.bind(ctx=mx.cpu(0), args=arg_arrays)

    exp = Experiment(NUM_ITERATIONS, NUM_IGNORED_ITERATIONS, EXPERIMENT_NAME)
    for i in range(NUM_ITERATIONS):
        with exp:
            output = executor.forward()
            out = output[0].wait_to_read()
        print("=" * 30)
        print("Finish an iteration %d" % i)
    exp.Summary()


def main():
    import sys
    global BATCH_SIZE
    global WEIGHT_SIZE
    global NUM_LAYERS
    global NUM_ITERATIONS
    global NUM_IGNORED_ITERATIONS
    global EXPERIMENT_NAME
    parser = argparse.ArgumentParser(description='Test p2pnet operators with '
                                                 'new Context implementation.')
    parser.add_argument('-a', '--addresses', type=str,
                        help='Addresses of all workers.')
    parser.add_argument('-f', '--host_file', type=str,
                        help='Host file that contains addresses of all workers.')
    parser.add_argument('-i', '--worker_index', type=int,
                        help='Index of this worker in addresses.')
    parser.add_argument('-s', '--single_machine', action='store_const',
                        const=True, help='Use single machine only.')
    parser.add_argument('-b', '--batch_size', type=int, help='Batch size',
                        default=BATCH_SIZE)
    parser.add_argument('-w', '--weight_size', type=int, help='Weight size',
                        default=WEIGHT_SIZE)
    parser.add_argument('-l', '--num_layers', type=int, help='Number of layers',
                        default=NUM_LAYERS)
    parser.add_argument('-n', '--num_iterations', type=int,
                        help='Number of ignored iteraions',
                        default=NUM_ITERATIONS)
    parser.add_argument('-g', '--num_ignored_iterations', type=int,
                        help='Number of ignored iterations when timing.',
                        default=NUM_IGNORED_ITERATIONS)
    parser.add_argument('-e', '--experiment_name', type=str,
                        help='The name of this experiment.',
                        default='Experiment')
    args = parser.parse_args()
    BATCH_SIZE = int(args.batch_size)
    WEIGHT_SIZE = int(args.weight_size)
    NUM_LAYERS = int(args.num_layers)
    NUM_ITERATIONS = int(args.num_iterations)
    NUM_IGNORED_ITERATIONS = int(args.num_ignored_iterations)
    EXPERIMENT_NAME = args.experiment_name
    if args.single_machine:
        Single()
    else:
        addresses = []
        if args.host_file is not None:
            with open(args.host_file) as fp:
                for line in fp:
                    if line.find(":") == -1:
                        addresses.append(line.strip() + ":9200")
                    else:
                        addresses.append(line.strip())
        else:
            addresses = args.addresses.split(',')
        if args.worker_index is not None:
            MLP_MP(addresses, int(args.worker_index))
        else:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            MLP_MP(addresses, comm.Get_rank())

if __name__ == "__main__":
    main()
