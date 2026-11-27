# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Blackwell state-backward launch restrictions and repeated-output regression."""

import ast
import inspect
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / 'fla/ops/common/chunk_delta_h.py'
CACHE = ROOT / 'fla/ops/utils/cache.py'


class Config:
    def __init__(self, kwargs, num_warps, num_stages, **extra):
        self.kwargs, self.num_warps, self.num_stages = kwargs, num_warps, num_stages
        self.__dict__.update(extra)


def load_policy():
    tree = ast.parse(COMMON.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_blackwell_dhu_cached_config_supported')
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(COMMON), 'exec'), namespace)
    return namespace[fn.name]


@pytest.mark.parametrize('blackwell', [False, True])
@pytest.mark.parametrize('large_smem', [False, True])
def test_backward_candidate_restriction(blackwell, large_smem):
    tree = ast.parse(COMMON.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == 'chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64')
    decorator = next(n for n in fn.decorator_list if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == 'fla_cache_autotune')
    namespace = dict(triton=SimpleNamespace(Config=Config), IS_NVIDIA_BLACKWELL=blackwell,
                     check_shared_mem=lambda arch: large_smem, _blackwell_dhu_cached_config_supported=load_policy())
    def evaluate(name):
        expr = next(k.value for k in decorator.keywords if k.arg == name)
        return eval(compile(ast.Expression(body=expr), str(COMMON), 'eval'), namespace)
    configs, policy = evaluate('configs'), evaluate('cached_config_filter')
    assert {c.num_warps for c in configs} == ({2} if blackwell else {2, 4})
    assert {c.num_stages for c in configs} == ({3} if blackwell else {2, 3, 4} if large_smem else {1})
    assert {c.kwargs['BV'] for c in configs} == ({32, 64} if large_smem else {32})
    if blackwell:
        assert all(policy(c) for c in configs)
    else:
        assert policy is None


@pytest.mark.parametrize('override,accepted', [
    ({}, True), ({'kwargs': {'BV': 64}}, True), ({'num_warps': 4}, False),
    ({'num_stages': 2}, False), ({'num_stages': 4}, False), ({'kwargs': {'BV': 128}}, False),
    ({'kwargs': {'BV': 32, 'EXTRA': 1}}, False), ({'num_ctas': 2}, False),
    ({'maxnreg': 32}, False), ({'ir_override': 'external.ptx'}, False),
])
@pytest.mark.parametrize('use_filter', [False, True])
def test_cache_cannot_restore_unsafe_launch(override, accepted, use_filter):
    class Autotuner:
        def __init__(self, fn, arg_names, configs, key, reset_to_zero, restore_value, **kwargs):
            assert 'cached_config_filter' not in kwargs
            self.cache = {}
            self.user_defined_pre_hook = self.user_defined_post_hook = False
            self.reset_to_zero, self.restore_value = reset_to_zero, restore_value

    entry = dict(kwargs={'BV': 32}, num_warps=2, num_stages=3, num_ctas=1)
    entry.update(override)
    tree = ast.parse(CACHE.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in ('CachedAutotuner', 'fla_cache_autotune')]
    namespace = dict(Autotuner=Autotuner, AutotuneKey=object, triton=SimpleNamespace(Config=Config),
                     TRITON_ABOVE_3_5_1=True, TRITON_ABOVE_3_4_0=True,
                     logger=logging.getLogger(__name__), load_cached_config=lambda *args: entry)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CACHE), 'exec'), namespace)
    policy = load_policy() if use_filter else None
    fn = SimpleNamespace(__name__='test_kernel', arg_names=[])
    tuner = namespace['fla_cache_autotune']([Config({'BV': 32}, 2, 3)], cached_config_filter=policy)(fn)
    key = SimpleNamespace(autotune_key=('shape',))
    tuner.cache[key.autotune_key] = 'previous_entry'
    tuner.maybe_load_cached_config(key)
    assert (key.autotune_key in tuner.cache) is (accepted or not use_filter)
    if accepted or not use_filter:
        assert tuner.cache[key.autotune_key].num_warps == entry['num_warps']


@pytest.mark.skipif(os.getenv('FLA_STATE_BWD_GPU_TEST') != '1', reason='explicit GPU opt-in')
@pytest.mark.parametrize('dtype_name,dim', [('float16', 64), ('bfloat16', 128)])
@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('gate', ['g', 'gk'])
def test_state_backward_repeat_and_layout(dtype_name, dim, packed, gate):
    import torch

    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
    from fla.utils import IS_NVIDIA_BLACKWELL, assert_close, device

    if not IS_NVIDIA_BLACKWELL:
        pytest.skip('Blackwell state-backward regression')
    torch.manual_seed(42)
    dtype = getattr(torch, dtype_name)
    b, t, h = (1 if packed else 4), 2048, 8
    def rand(shape):
        return torch.randn(shape, dtype=dtype, device=device) * .05
    cu = torch.tensor([0, 63, 511, 2048], device=device) if packed else None
    n = 3 if packed else b
    args = {name: rand((b, t, h, dim)) for name in ('q', 'k', 'w', 'do', 'dv')}
    args[gate] = -torch.rand((b, t, h) + ((dim,) if gate == 'gk' else ()), device=device)
    args.update(h0=rand((n, h, dim, dim)).float(), dht=rand((n, h, dim, dim)).float(),
                scale=.1, cu_seqlens=cu, chunk_size=64)
    # exercise the shared Triton kernel even if another backend is enabled
    fn = inspect.unwrap(chunk_gated_delta_rule_bwd_dhu)
    expected = fn(**args)
    snapshots = {key: value.clone() for key, value in args.items() if isinstance(value, torch.Tensor)}
    for _ in range(3):
        actual = fn(**args)
        for ref, got in zip(expected, actual):
            assert torch.isfinite(got).all()
            torch.testing.assert_close(ref, got, atol=0, rtol=0)
    transposed = dict(args, state_v_first=True, h0=args['h0'].transpose(-1, -2).contiguous(),
                      dht=args['dht'].transpose(-1, -2).contiguous())
    actual = fn(**transposed)
    for index, (ref, got) in enumerate(zip(expected, actual)):
        if index < 2:
            got = got.transpose(-1, -2)
        assert_close(('dh', 'dh0', 'dv')[index], ref.float(), got.float(), 1e-4)
    for key, before in snapshots.items():
        torch.testing.assert_close(before, args[key], atol=0, rtol=0)
