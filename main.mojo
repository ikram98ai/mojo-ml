from gpu.host import DeviceContext
from sys import has_accelerator

def main():
    num_float16 = SIMD[DType.float16, 4](3.5, -3.5, 3.5, -3.5)
    denom_float16 = SIMD[DType.float16, 4](2.5, 2.5, -2.5, -2.5)

    num_int32 = SIMD[DType.int32, 4](5, -6, 7, -8)
    denom_int32 = SIMD[DType.int32, 4](2, 3, -4, -5)

    # Result is SIMD[DType.float16, 4]
    true_quotient_float16 = num_float16 / denom_float16
    print("True float16 division:", true_quotient_float16)

    # Result is SIMD[DType.int32, 4]
    true_quotient_int32 = num_int32 / denom_int32
    print("True int32 division:", true_quotient_int32)

    # Result is SIMD[DType.float16, 4]
    var floor_quotient_float16 = num_float16 // denom_float16
    print("Floor float16 division:", floor_quotient_float16)

    # Result is SIMD[DType.int32, 4]
    var floor_quotient_int32 = num_int32 // denom_int32
    print("Floor int32 division:", floor_quotient_int32)


    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        ctx = DeviceContext()
        print("Found GPU:", ctx.name())