# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from typing import Any

from nias.exceptions import InversionError
from nias.interfaces import LinearOperator, LinearSolver, LinearSolverFactory


class DefaultLinearSolverFactory(LinearSolverFactory):

    def __init__(self):
        self.solvers = []
        self.defaults = {}

    def register_solver(self, supported_types: tuple[type], name: str,
                        solver: type[LinearSolver], defaults: dict[str, Any]):
        self.solvers.append((supported_types, name, solver))
        self.defaults[name] = defaults

    def get_solver(self, lhs: LinearOperator, context: str = '') -> LinearSolver:
        from nias.base.operators import LincombOperator

        if isinstance(lhs, LincombOperator):
            base_types = tuple(type(o) for o in lhs.operators)
        else:
            base_types = (type(lhs),)

        for supported_types, name, solver in self.solvers:
            if all(isinstance(bt in supported_types) for bt in base_types):
                return solver(lhs, **self.defaults[name])

        raise InversionError(f'No solver known for base types {base_types}')

    def set_defaults(self, name: str, **values) -> None:
        self.defaults[name].update(values)


default_factory = DefaultLinearSolverFactory()
