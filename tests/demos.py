# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import importlib
from pathlib import Path

import pytest
from typer import Typer
from typer.testing import CliRunner

DEMO_ARGS = (
    ('eigs', [10]),
    ('gram_schmidt', [10, 5]),
    ('svd', [10, 4]),
)


runner = CliRunner()

def _demo_ids(demo_args):
    return ['_'.join([args[0]] + [str(a) for a in args[1]]) for args in demo_args]


@pytest.fixture(params=DEMO_ARGS, ids=_demo_ids(DEMO_ARGS))
def demo_args(request):
    return request.param


def test_demos(demo_args):
    script, args = demo_args

    # Create a spec for the module
    base_path = Path(__file__).parent.parent / 'demos'
    spec = importlib.util.spec_from_file_location(script, base_path / f'{script}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    app = Typer()
    app.command()(module.main)
    result = runner.invoke(app, [str(arg) for arg in args], catch_exceptions=False)
    assert result.exit_code == 0
