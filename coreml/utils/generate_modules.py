from functions import *

if __name__ == "__main__":
    MODULES_PATH = "../modules/"
    # gen_identity(128, MODULES_PATH + "identity_128.mlpackage")

    # gen_matmul_transpose({
    #     "lower_bound": 1,
    #     "upper_bound": 256,
    #     "default": 128,
    # }, 128, MODULES_PATH + "matmul_transpose_128.mlpackage")

    # gen_softmax(MODULES_PATH + "softmax_4096.mlpackage")

    # gen_normalize(MODULES_PATH + "llama_rmsnorm_4096.mlpackage")

    gen_static_matmul_transpose(1, 32000, 4096, MODULES_PATH +  "lm_head.mlpackage")
    gen_static_matmul_transpose(1, 4096, 4096, MODULES_PATH +  "QKV_out_proj.mlpackage")
    gen_static_matmul_transpose(1, 4096, 11008, MODULES_PATH +  "down_proj.mlpackage")
    gen_static_matmul_transpose(1, 11008, 4096, MODULES_PATH +  "gate_proj.mlpackage")