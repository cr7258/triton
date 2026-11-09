import pytest
import torch

import triton
import triton.language as tl


@triton.jit
def _rms_kernel(x_ptr, y_ptr, N: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    offsets = row * N + tl.arange(0, N)
    x = tl.load(x_ptr + offsets).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / N
    tl.store(y_ptr + offsets, x * tl.rsqrt(mean_square + EPS))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n", [128, 1024])
def test_sm107_reduction(dtype, n):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("Test requires a CUDA target.")
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Test requires SM107.")

    x = torch.randn((8, n), device="cuda", dtype=dtype)
    y = torch.empty_like(x)
    eps = 1e-6
    kernel = _rms_kernel[(x.shape[0], )](x, y, n, eps, num_warps=4)

    reference = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(y, reference.to(dtype), rtol=tolerance, atol=tolerance)
    assert ".target sm_107a" in kernel.asm["ptx"]
