from matmul_transpose import gen_matmul_transpose

MODULES_PATH = "../modules/"

if __name__ == "__main__":
    gen_matmul_transpose({
        "lower_bound": 1,
        "upper_bound": 256,
        "default": 128,
    }, 128, MODULES_PATH + "matmul_transpose_128.mlpackage")