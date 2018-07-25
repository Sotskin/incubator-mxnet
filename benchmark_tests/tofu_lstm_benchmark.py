# pylint: skip-file
from __future__ import print_function

from collections import namedtuple
import os, sys,time
import pickle as pickle
import logging
import argparse

import numpy as np
import mxnet as mx

rng = np.random.RandomState(seed=42)

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "h2h_weight"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx):
    """LSTM Cell symbol"""
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                no_bias=True,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                no_bias=True,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)

def get_symbol(args):
    param_cells = []
    last_states = []

    for i in range(args.num_layers):
        with mx.AttrScope(ctx_group='layer%d' % i):
            param_cells.append(LSTMParam(i2h_weight = mx.sym.Variable("l%d_i2h_weight" % i),
                                         h2h_weight = mx.sym.Variable("l%d_h2h_weight" % i)))
            state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                              h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    with mx.AttrScope(ctx_group='output'):
        cls_weight = mx.sym.Variable("cls_weight")
    assert(len(last_states) == args.num_layers)

    logits = []
    for seqidx in range(args.seq_len):
        with mx.AttrScope(ctx_group='data'):
            hidden = mx.sym.Variable("data%d" % seqidx)
        # stack LSTM
        for i in range(args.num_layers):
            with mx.AttrScope(ctx_group='layer%d' % i):
                next_state = lstm(args.hidden_size, indata=hidden,
                                  prev_state=last_states[i],
                                  param=param_cells[i],
                                  seqidx=seqidx, layeridx=i)
                hidden = next_state.h
                last_states[i] = next_state
        with mx.AttrScope(ctx_group='output'):
            pred = mx.sym.FullyConnected(data=hidden,
                                         weight=cls_weight,
                                         no_bias=True,
                                         num_hidden=args.num_classes,
                                         name='logits%d' % i)
            logits.append(mx.sym.SoftmaxOutput(pred))
    return mx.sym.Group(logits)

def feed_args(net, arg_arrays):
    names = net.list_arguments()
    for name, arr in zip(names, arg_arrays):
        if not name.endswith('label'):
            arr[:] = 0.0

def test():
    # print logging by default
    mx.base.start_iteration()
    logging.basicConfig(level=logging.DEBUG)
    print(sys.argv)
    parser = argparse.ArgumentParser("Tofu GPU lstm test")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=30, help='Sequence length')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of lstm layers')
    parser.add_argument('--hidden_size', type=int, default=1024, help='Size of lstm layers')
    parser.add_argument('--input_size', type=int, default=512, help='Size of input embeddings')
    parser.add_argument('--num_classes', type=int, default=512, help='Number of output classes')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--num_loops', type=int, default=30, help='Number of benchmarking loops.')
    parser.add_argument('--cold_skip', type=int, default=5, help='Number of loops skipped for warm up.')
    parser.add_argument('--use_momentum', type=int, default=0, help='Whether to simulate memory consumption with momentum.')

    args, _ = parser.parse_known_args()

    num_loops = args.num_loops
    cold_skip = args.cold_skip

    print(args)
    net = get_symbol(args)
    print(net)

    print(net.list_arguments())
    print(net.list_outputs())

    #group2ctx = {'group:%d' % i : mx.gpu(i) for i in range(args.num_gpus)}
    group2ctx = {}
    group2ctx['data'] = mx.gpu(0)
    group2ctx['output'] = mx.gpu(args.num_gpus - 1)
    for i in range(args.num_layers):
        group2ctx['layer%d' % i] = mx.gpu(i // (args.num_layers // args.num_gpus))
    print(group2ctx)
    default_ctx = mx.cpu(0)

    name2ctx = {}
    in_shapes = {}
    in_types = {}
    for i in range(args.seq_len):
        in_shapes['data%d' % i] = (args.batch_size, args.input_size)
        in_types['data%d' % i] = mx.base.mx_real_t
        name2ctx['data%d' % i] = group2ctx['data']
        name2ctx['softmaxoutput%d_label' % i] = group2ctx['output']
    for i in range(args.num_layers):
        in_shapes['l%d_init_c' % i] = (args.batch_size, args.hidden_size)
        in_types['l%d_init_c' % i] = mx.base.mx_real_t
        in_shapes['l%d_init_h' % i] = (args.batch_size, args.hidden_size)
        in_types['l%d_init_h' % i] = mx.base.mx_real_t
        name2ctx['l%d_init_c' % i] = group2ctx['layer%d' % i]
        name2ctx['l%d_init_h' % i] = group2ctx['layer%d' % i]
        name2ctx['l%d_i2h_weight' % i] = group2ctx['layer%d' % i]
        name2ctx['l%d_h2h_weight' % i] = group2ctx['layer%d' % i]
    name2ctx['cls_weight'] = group2ctx['output']
    arg_shapes, out_shapes, aux_shapes = net.infer_shape(**in_shapes)
    arg_types, out_types, aux_types = net.infer_type(**in_types)

    # create ndarrays for all arguments.
    arg_arrays = [mx.nd.zeros(shape, name2ctx[name], dtype=dtype)
                  for name, shape, dtype in zip(net.list_arguments(), arg_shapes, arg_types)]
    print('Num arguments: ', len(arg_arrays))
    # create gradient ndarray for all parameters.
    args_grad = {name : mx.nd.zeros(shape, name2ctx[name], dtype=dtype)
                 for name, shape, dtype in zip(net.list_arguments(), arg_shapes, arg_types)
                 if not name.startswith('data') and not name.endswith('label')}
    print('Argument grads: ', args_grad.keys())
    if args.use_momentum:
        assert False, "Momentum simulation is not required since this MXNet implementation \
                is not able to do fully inplace gradient computation. An extra gradient \
                buffer is used which ends up with the same memory consumption as maintaining \
                a momentum."
        args_mom = {name : mx.nd.zeros(shape, name2ctx[name], dtype=dtype)
                    for name, shape, dtype in zip(net.list_arguments(), arg_shapes, arg_types)
                    if not name.startswith('data') and not name.endswith('label')}

    executor = net.bind(ctx=default_ctx,
                        args=arg_arrays,
                        args_grad=args_grad,
                        grad_req='write',
                        group2ctx=group2ctx)
    feed_args(net, arg_arrays)
    import time
    time.sleep(10)
    mx.base.stop_iteration()
    print("Bind End")
    all_time = []
    for i in range(num_loops):
        print('=> loop %d' % i);
        mx.base.start_iteration()
        st_l = time.time()
        if i == cold_skip + 1:
            t0 = time.time()
        outputs = executor.forward()
        executor.backward()
        for name, grad in args_grad.items():
            #print(name, grad.asnumpy())
            grad.wait_to_read()
        mx.base.stop_iteration()
        ed_l = time.time()
        print('=> loop duration %f' % float(ed_l - st_l))
        if (i >= cold_skip):
             all_time.append(float(ed_l - st_l))
    t1 = time.time()

    duration = t1 - t0
    print('duration %f, average %f, std %f' % \
        (duration, np.asarray(all_time).mean(), np.asarray(all_time).std()))

if __name__ == "__main__":
    print('================ Test Begin ====================')
    test()
    print('================ Test Finished ====================')
