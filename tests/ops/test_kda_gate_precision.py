# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Mixed-precision gate parameter gradients must reduce the FP32 kernel output."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('has_bias', [False, True])
@pytest.mark.parametrize('has_a', [False, True])
def test_gate_bwd_parameter_reduction(dtype, has_bias, has_a):
    # execute the actual host wrapper with a synthetic kernel output, without a GPU
    source = Path(__file__).resolve().parents[2] / 'fla/ops/kda/gate.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'kda_gate_bwd')
    function.decorator_list = []
    g = torch.zeros(1, 65, 2, 3, dtype=dtype)
    derivative = torch.ones_like(g, dtype=torch.float32)
    derivative[:, :64:2] = 1.0002
    derivative[:, 1:64:2] = -1.0
    a = torch.zeros(1, 1, 2, 1) if has_a else None
    bias = torch.zeros(6) if has_bias else None

    class Kernel:
        def __getitem__(self, grid):
            def launch(**kwargs):
                kwargs['dg'].copy_(derivative)
                if kwargs['dA'] is not None:
                    kwargs['dA'].fill_(0.125)
            return launch

    namespace = dict(torch=torch, kda_gate_bwd_kernel=Kernel(),
                     triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b,
                                            next_power_of_2=lambda n: 1 << (n - 1).bit_length()))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    dg, da, dbias = namespace['kda_gate_bwd'](g=g, A_log=a, dt_bias=bias, dyg=derivative, lower_bound=-5.)
    torch.testing.assert_close(dg, derivative.to(dtype), rtol=0, atol=0)
    if has_a:
        torch.testing.assert_close(da, torch.full_like(a, 0.375), rtol=0, atol=0)
    else:
        assert da is None
    if has_bias:
        torch.testing.assert_close(dbias, derivative.sum((0, 1)).flatten(), rtol=0, atol=0)
    else:
        assert dbias is None


@pytest.mark.skipif(os.getenv('FLA_KDA_GATE_PRECISION_GPU_TEST') != '1', reason='explicit GPU opt-in')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('lower_bound', [None, -5.0])
@pytest.mark.parametrize('cancellation', [False, True])
def test_fused_gate_mixed_precision_parameters(dtype, lower_bound, cancellation):
    from fla.ops.kda.gate import fused_kda_gate, naive_kda_gate, naive_kda_lowerbound_gate
    from fla.utils import assert_close, device

    torch.manual_seed(42)
    shape = (2, 257, 3, 60)
    g = torch.randn(shape, dtype=dtype, device=device)
    a = torch.randn(3, device=device) * .1
    bias = torch.randn(180, device=device) * .1
    dy = torch.randn(shape, dtype=torch.float32, device=device)
    if cancellation:
        g.zero_()
        a.zero_()
        bias.zero_()
        dy[:, :256:2] = 2.0004
        dy[:, 1:256:2] = -2.0
        dy[:, 256] = 0.0
    inputs = tuple(x.requires_grad_() for x in (g, a, bias))
    reference_fn = naive_kda_gate if lower_bound is None else naive_kda_lowerbound_gate
    kwargs = {} if lower_bound is None else {'lower_bound': lower_bound}
    ref = reference_fn(g=g, A_log=a, dt_bias=bias, **kwargs)
    actual = fused_kda_gate(g=g, A_log=a, dt_bias=bias, lower_bound=lower_bound)
    reference_grads = torch.autograd.grad(ref, inputs, dy)
    actual_grads = torch.autograd.grad(actual, inputs, dy)
    assert_close('o', ref, actual, 1e-4)
    for name, expected, result in zip(('dg', 'dA', 'dbias'), reference_grads, actual_grads):
        assert result.dtype == expected.dtype
        assert torch.isfinite(result).all()
        assert_close(name, expected.float(), result.float(), 1e-4)
