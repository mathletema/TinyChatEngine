import torch
from torch import nn

import coremltools as ct
from coremltools.converters.mil.mil import Builder, Function, Program
from coremltools.converters.mil.backend.mil.load import load

def gen_softmax(out_path):
    class SoftMax(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.softmax(x, dim=-1)

    input_shape = ct.Shape(shape=(
        32,
        ct.RangeDim(
            lower_bound=1,
            upper_bound=4096,
            default=64 ),
        ct.RangeDim(
            lower_bound=1,
            upper_bound=4096,
            default=64 )
    ))

    example_x = torch.rand(32, 64, 64)

    traced_model = torch.jit.trace(SoftMax().eval(),[example_x,])

    model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(shape=input_shape, name="x"),
        ],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram",
    )

    model.save(out_path)