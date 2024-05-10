# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import importlib
import pytest


DEMO_ARGS = (
    ('gram_schmidt', []),
    ('eigs', []),
)


@pytest.fixture(params=DEMO_ARGS)
def demo_args(request):
    return request.param


def test_demos(demo_args):
    module, args = demo_args

    # Create a spec for the module
    module_spec = importlib.util.spec_from_file_location('__main__', f'demos/{module}.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
