import os
import sys
import tensorflow as tf
import subprocess
from tensorflow.python.framework import ops

# Register ops for compilation here
OP_NAMES = ['correlation', 'decode_flo', 'decode_ppm']

cwd = os.getcwd()
current_file_dir = os.path.dirname(os.path.realpath(__file__))

dynamic_lib_dir = 'dynamic_libs/'
dynamic_lib_dir = current_file_dir + '/' + dynamic_lib_dir

source_code_dir = 'source_code/'
source_code_dir = current_file_dir + '/' + source_code_dir

os.chdir(source_code_dir)

if not os.path.isdir(dynamic_lib_dir):
    os.mkdir(dynamic_lib_dir)

def compile(op=None):
    if op is not None:
        to_compile = [op]
    else:
        to_compile = OP_NAMES

    tf_cflags = " ".join(tf.sysconfig.get_compile_flags())
    tf_lflags = " ".join(tf.sysconfig.get_link_flags())
    for n in to_compile:

        print('\n\ncompiling custom operation: ' + n + '\n\n')

        base = n + "_op"
        fn_cu_cc = base + ".cu.cc"
        fn_cc = base + ".cc"
        fn_cu_o = dynamic_lib_dir + base + ".cu.o"
        fn_so = dynamic_lib_dir + base + ".so"

        out, err = subprocess.Popen(['which', 'nvcc'], stdout=subprocess.PIPE).communicate()
        cuda_dir = out.decode().split('/cuda')[0]


        if os.path.isfile(os.getcwd() + '/' + fn_cu_cc):
            nvcc_cmd = "nvcc -std=c++11 -c -o {} {} {} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -I " + cuda_dir + " --expt-relaxed-constexpr"
            nvcc_cmd = nvcc_cmd.format(" ".join([fn_cu_o, fn_cu_cc]), tf_cflags, tf_lflags)
            subprocess.check_output(nvcc_cmd, shell=True)

            gcc_cmd = "{} -std=c++11 -shared -o {} {} -fPIC -L " + cuda_dir + "/cuda/lib64 -lcudart {} -O2 -D GOOGLE_CUDA=1"
            gcc_cmd = gcc_cmd.format('g++'," ".join([fn_so, fn_cu_o, fn_cc]), tf_cflags, tf_lflags)
        else:
            gcc_cmd = "{} -std=c++11 -shared {} -o {} -fPIC {} {} -O2"
            gcc_cmd = gcc_cmd.format('g++', fn_cc, fn_so, tf_cflags, tf_lflags)
            print('gcc_cmd: ' + gcc_cmd)
        subprocess.check_output(gcc_cmd, shell=True)


module = sys.modules[__name__]
for n in OP_NAMES:
    lib_path = './{}_op.so'.format(n)
    try:
        os.chdir(dynamic_lib_dir)
        op_lib = tf.load_op_library(lib_path)
    except:
        os.chdir(source_code_dir)
        compile(n)
        os.chdir(dynamic_lib_dir)
        op_lib = tf.load_op_library(lib_path)
    setattr(module, '_' + n + '_module', op_lib)

os.chdir(cwd)

# functions #
#-----------#

def correlation(first, second, **kwargs):
    return _correlation_module.correlation(first, second, **kwargs)[0]

decode_flo = _decode_flo_module.decode_flo
decode_ppm = _decode_ppm_module.decode_ppm

# Register op gradients

@ops.RegisterGradient("Correlation")
def _CorrelationGrad(op, in_grad, in_grad1, in_grad2):
    grad0, grad1 = _correlation_module.correlation_grad(
        in_grad, op.inputs[0], op.inputs[1],
        op.outputs[1], op.outputs[2],
        kernel_size=op.get_attr('kernel_size'),
        max_displacement=op.get_attr('max_displacement'),
        pad=op.get_attr('pad'),
        stride_1=op.get_attr('stride_1'),
        stride_2=op.get_attr('stride_2'))
    return [grad0, grad1]

ops.NotDifferentiable("DecodeFlo")
ops.NotDifferentiable("DecodePpm")