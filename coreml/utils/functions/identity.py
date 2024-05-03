import torch
from torch import nn

import coremltools as ct
from coremltools.converters.mil.mil import Builder, Function, Program
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.backend.mil.load import load

def gen_identity(dim_x, out_path):
    class Identity(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x

    input_shape = ct.Shape(
        shape=(dim_x,)
    )

    example_x = torch.rand(dim_x)

    traced_model = torch.jit.trace(Identity().eval(), [example_x])

    model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(shape=input_shape, name="x"),
        ],
        outputs=[ct.TensorType(name="x")],
        convert_to="mlprogram",
    )

    model.save(out_path)

def gen_static_identity(m, n, k, out_path):
    @mb.program(input_specs=[mb.TensorSpec(shape=(m, k)), mb.TensorSpec(shape=(n, k))])
    def prog(x, y):
        return x
    
    print(prog)

    model = ct.convert(prog)

    model.save(out_path)