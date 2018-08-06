'''
Usage:
Here, we use resnet as an example.
1. # python3 run.py resnet --dump
    * Modify config_resnet to fit your environment and requirement.
2. # python3 run.py @config_resnet
'''
import argparse
import os
import pprint
import signal
import subprocess
import sys


def run_script(args):
    if args.model == 'resnet':
        log_name = './log_{}_{}_{}_{}_{}_{}_{}'.format(
                        args.model, args.num_layers, args.batch_size,
                        args.wide_scale, args.mxnet_swap_algorithm,
                        args.mxnet_prefetch_algorithm,
                        args.mxnet_prefetch_steps)
        options = ['--num_gpus={}'.format(args.num_gpus),
                   '--num_layers={}'.format(args.num_layers),
                   '--wide_scale={}'.format(args.wide_scale),
                   '--batch_size={}'.format(args.batch_size)]
    elif args.model == 'rnn':
        log_name = 'TBD'
        options = []
        raise NotImplementedError
    else:
        raise NotImplementedError
    options.append(args.model)
    envs = {arg.upper(): str(getattr(args, arg)) for arg in vars(args)}
    proc = subprocess.Popen(['python', 'benchmark.py'] + options, env=envs,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=1, universal_newlines=True)
    with open(log_name, 'w') as fp:
        while True:
            line = proc.stdout.readline()
            if line:
                if args.also_print:
                    print(line, end='')
                fp.write(line)
            else:
                break
        stdout, stderr = proc.communicate()
        message = 'The program exits with returncode = {}\n'\
                    .format(proc.returncode)
        if proc.returncode != 0:
            name = 'None'
            for k, v in signal.__dict__.items():
                if k.startswith('SIG') and not k.startswith('SIG_'):
                    if getattr(signal, k) == -proc.returncode:
                        name = k
                        break
            if name == 'SIGSEGV':
                message += 'Segmentation Fault.\n'
            else:
                message += name + " : " + str(proc.returncode) + '\n'
        fp.write(message)
        print(message)


class DumpArguments(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        nargs=0
        super(DumpArguments, self).__init__(option_strings, dest, nargs=nargs,
                                            **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        path = os.path.join('.', 'config_' + namespace.model)
        if os.path.exists(path):
            print('{} will be overwitten. '
                  'Do you want to continue? (y/N) '.format(path))
            answer = sys.stdin.readline().strip().lower()
            if answer != 'y':
                sys.exit(0)
        with open(path, 'w') as fp:
            fp.write(getattr(namespace, 'model') + '\n')
            for arg in sorted(vars(namespace)):
                if arg != 'model' and arg != 'dump':
                    val = getattr(namespace, arg)
                    if type(val) == bool:
                        fp.write('--' + ('' if val else 'no') +  arg + '\n')
                    else:
                        fp.write('--' + arg + '\n')
                        fp.write(str(val) + '\n')
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser('Benchmark Tests',
                                     fromfile_prefix_chars='@')
    parser.add_argument('model', type=str, default='resnet',
                        help='The model to be tested.')
    parser.add_argument('--dump', type=bool, action=DumpArguments)
    parser.add_argument('--also_print', action='store_true')
    # Environment settings
    parser.add_argument('--pythonpath', type=str, default='/home/fegin/',
                        help='PYTHONPATH')
    parser.add_argument('--ld_library_path', type=str, default='/usr/local/cuda/lib64',
                        help='Extra LD_LIBRARY_PATH')
    parser.add_argument('--cuda_visible_devices', type=int, default=0,
                        help='CUDA_VISIBLE_DEVICE')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs.')
    parser.add_argument('--num_loops', type=int, default=10,
                        help='Number of loops.')
    # Model settings 
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--num_layers', type=int, default=152,
                        help='Number of layers (for some models only).')
    parser.add_argument('--wide_scale', type=int, default=1,
                        help='Wide scale (for W-ResNet only).')
    # Swap settings
    parser.add_argument('--mxnet_engine_type', type=str, default='NaiveEngine',
                        help='MXNET_ENGINE_TYPE')
    parser.add_argument('--mxnet_prefetch_algorithm', type=str,
                        default='ComputePrefetch',
                        help='Prefetch look ahead steps.')
    parser.add_argument('--mxnet_mem_mgr_type', type=str, default='CUDA',
                        help='The memory manager type for the swapping system.')
    parser.add_argument('--mxnet_prefetch_steps', type=int, default=30,
                        help='Prefetch look ahead steps.')
    parser.add_argument('--mxnet_swap_algorithm', type=str,
                        default='NaiveHistory',
                        help='Swap algorithm.')

    args, _ = parser.parse_known_args()
    print('Arguments: ')
    for arg in sorted(vars(args)):
        print('    {}: {}'.format(arg, str(getattr(args, arg))))

    run_script(args)

if __name__ == '__main__':
    main()


