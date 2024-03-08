import torch
from torch import nn

import coremltools as ct
from coremltools.converters.mil.mil import Builder, Function, Program
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.backend.mil.load import load

def gen_static_matmul_transpose(m, n, k, out_path):
    @mb.program(input_specs=[mb.TensorSpec(shape=(m, k)), mb.TensorSpec(shape=(n, k))])
    def prog(x, y):
        return mb.matmul(x=x, y=y, transpose_x=False, transpose_y=True, name="matmul")
    
    print(prog)

    model = ct.convert(prog)

    model.save(out_path)