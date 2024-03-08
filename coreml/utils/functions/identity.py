import torch
from torch import nn

import coremltools as ct
from coremltools.converters.mil.mil import Builder, Function, Program
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