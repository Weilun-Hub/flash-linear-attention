# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Standalone launch-policy tests and opt-in CUDA candidate parity tests."""

import importlib.util
import os
from pathlib import Path

import pytest


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TUNING = load_file('_wy_tuning', Path(__file__).resolve().parents[2] /
                   'fla/ops/kda/backends/tilelang/tuning.py')


def benchmark_path():
    override = os.environ.get('FLA_KDA_BWD_BENCHMARK_PATH')
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'FLA_KDA_BWD_BENCHMARK_PATH is not a file: {path}')
        return path
    # The active FLA checkout may live outside DataAgent or be reached through
    # a symlink. Search the invocation directory as well as the test ancestors.
    cwd = Path.cwd().resolve()
    for root in (cwd, *cwd.parents, *Path(__file__).resolve().parents):
        path = root / 'workspace/scripts/benchmark_kda_backward.py'
        if path.is_file():
            return path
    raise FileNotFoundError(
        'Cannot find benchmark_kda_backward.py. Run from the DataAgent repository '
        'or set FLA_KDA_BWD_BENCHMARK_PATH to its absolute file path.'
    )


@pytest.mark.parametrize('location', ['override', 'cwd', 'test_ancestor'])
def test_benchmark_discovery_with_separate_checkout(monkeypatch, tmp_path, location):
    module = __import__(__name__, fromlist=['benchmark_path'])
    project = tmp_path / 'DataAgent'
    external = tmp_path / 'external-fla/tests/ops/test_tuning.py'
    monkeypatch.setattr(module, '__file__', str(external))
    monkeypatch.delenv('FLA_KDA_BWD_BENCHMARK_PATH', raising=False)
    monkeypatch.chdir(tmp_path)
    target = project / 'workspace/scripts/benchmark_kda_backward.py'
    target.parent.mkdir(parents=True)
    target.write_text('MARKER = 42\n')
    if location == 'override':
        monkeypatch.setenv('FLA_KDA_BWD_BENCHMARK_PATH', str(target))
    elif location == 'cwd':
        monkeypatch.chdir(target.parent)
    else:
        monkeypatch.setattr(module, '__file__', str(project / 'runtime/fla/tests/ops/test_tuning.py'))
    assert benchmark_path() == target
    assert load_file('_benchmark_discovery_check', benchmark_path()).MARKER == 42


@pytest.mark.parametrize('explicit', [False, True])
def test_missing_benchmark_has_actionable_error(monkeypatch, tmp_path, explicit):
    module = __import__(__name__, fromlist=['benchmark_path'])
    monkeypatch.setattr(module, '__file__', str(tmp_path / 'external/tests/ops/test_tuning.py'))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('FLA_KDA_BWD_BENCHMARK_PATH', raising=False)
    if explicit:
        monkeypatch.setenv('FLA_KDA_BWD_BENCHMARK_PATH', str(tmp_path / 'missing.py'))
    with pytest.raises(FileNotFoundError, match='FLA_KDA_BWD_BENCHMARK_PATH'):
        benchmark_path()


def test_unset_preserves_default(monkeypatch):
    monkeypatch.delenv('FLA_KDA_BWD_WY_TILE', raising=False)
    monkeypatch.delenv('FLA_KDA_BWD_WY_HEAD_FIRST', raising=False)
    assert TUNING.backward_launch_config(128, 128, 64, 64, 4) == (64, 64, 4, False)


@pytest.mark.parametrize('tile', ['64,64,4', '64,64,8', '32,64,4', '64,32,4', '128,128,8'])
@pytest.mark.parametrize('order', [0, 1])
def test_explicit_configuration(monkeypatch, tile, order):
    monkeypatch.setenv('FLA_KDA_BWD_WY_TILE', tile)
    monkeypatch.setenv('FLA_KDA_BWD_WY_HEAD_FIRST', str(order))
    assert TUNING.backward_launch_config(128, 128, 64, 64, 4) == (*map(int, tile.split(',')), bool(order))


@pytest.mark.parametrize('tile,k,v', [
    ('64,64', 128, 128), ('64,64,4,8', 128, 128), ('64,64,a', 128, 128),
    ('16,64,4', 128, 128), ('64,64,2', 128, 128), ('64,64,4', 96, 128),
    ('64,64,4', 128, 96), ('128,128,8', 64, 128),
])
def test_unsafe_tile_rejected(monkeypatch, tile, k, v):
    monkeypatch.setenv('FLA_KDA_BWD_WY_TILE', tile)
    monkeypatch.setenv('FLA_KDA_BWD_WY_HEAD_FIRST', '0')
    with pytest.raises(ValueError):
        TUNING.backward_launch_config(k, v, 64, 64, 4)


@pytest.mark.skipif(os.getenv('FLA_KDA_BWD_TUNING_GPU_TEST') != '1', reason='explicit CUDA/TileLang test opt-in')
@pytest.mark.parametrize('tile,order', [('64,64,4', 1), ('64,64,8', 0), ('32,64,4', 1), ('64,32,4', 1)])
@pytest.mark.parametrize('layout,k,v,state_v_first', [
    ('dense', 128, 128, False), ('packed', 128, 128, False),
    ('packed', 64, 128, True), ('dense', 128, 64, True),
])
def test_tilelang_candidate_parity(monkeypatch, tile, order, layout, k, v, state_v_first):
    import torch
    from fla.utils import IS_NVIDIA
    if not IS_NVIDIA:
        pytest.skip('TileLang launch candidates require NVIDIA')
    from fla.ops.kda.backends.tilelang.chunk_bwd_dqkg import chunk_kda_bwd_wy_dqkg_fused_tilelang as fn
    bench = load_file('_wy_benchmark', benchmark_path())
    torch.manual_seed(42)
    lengths = [1, 62, 129, 65] if layout == 'packed' else None
    kwargs = bench.make_inputs(torch, 257, 3, k, v, lengths, state_v_first,
                               batch=1 if lengths else 2)
    # Exercise full writes independently of repository-wide NaN fixtures.
    original = torch.empty_like
    def poisoned(*args, **kw):
        out = original(*args, **kw)
        return out.fill_(float('nan')) if out.is_floating_point() else out
    monkeypatch.setattr(torch, 'empty_like', poisoned)
    with bench.tuning():
        expected = fn(**kwargs)
    with bench.tuning(tile, order):
        actual = fn(**kwargs)
    torch.cuda.synchronize()
    bench.validate(torch, expected, actual)
