import torch
from torch import nn

import coremltools as ct
from coremltools.converters.mil.mil import Builder, Function, Program
from coremltools.converters.mil.backend.mil.load import load

def gen_normalize(out_path):
    class RMSNorm(nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x, weight):
            var_x = torch.sqrt(torch.mean(x * x, dim=2, keepdims=True) + 0.00001)
            x /= var_x
            return weight * x
        
    x_shape = ct.Shape(shape=(
        1,
        ct.RangeDim(
            lower_bound=1,
            upper_bound=4096,
            default=64 ),
        4096
    ))

    weight_shape = ct.Shape(shape=(1, 1, 4096))

    example_x = torch.rand(1, 64, 4096)
    example_weight = torch.randn(1, 1, 4096)

    traced_model = torch.jit.trace(RMSNorm().eval(), [example_x, example_weight])

    model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(shape=x_shape, name="x"),
            ct.TensorType(shape=weight_shape, name="weight")
        ],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram",
    )

    model.save(out_path)