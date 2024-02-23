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

def gen_matmul_transpose(dim_x_range, dim_y, out_path):
    class BMM(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.matmul(x, torch.transpose(y, -1, -2))

    input_shape = ct.Shape(
        shape=(ct.RangeDim(
            lower_bound=dim_x_range["lower_bound"],
            upper_bound=dim_x_range["upper_bound"],
            default=dim_x_range["default"]
        ), 128)
    )

    example_a = torch.rand(dim_x_range["default"], dim_y)
    example_b = torch.rand(dim_x_range["default"], dim_y)

    traced_model = torch.jit.trace(BMM().eval(), [example_a, example_b])

    model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(shape=input_shape, name="x"),
            ct.TensorType(shape=input_shape, name="y"),
        ],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram",
    )

    model.save(out_path)

if __name__ == "__main__":
    MODULES_PATH = "../modules/"
    gen_identity(128, MODULES_PATH + "identity_128.mlpackage")

    gen_matmul_transpose({
        "lower_bound": 1,
        "upper_bound": 256,
        "default": 128,
    }, 128, MODULES_PATH + "matmul_transpose_128.mlpackage")