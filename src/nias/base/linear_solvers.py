# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from typing import Any

from nias.exceptions import InversionError
from nias.interfaces import LinearOperator, LinearSolver, LinearSolverFactory


class DefaultLinearSolverFactory(LinearSolverFactory):

    solvers: list[tuple[int, tuple[type], str, type[LinearSolver]]]

    def __init__(self):
        self.solvers = []
        self.defaults = {}

    def register_solver(self, supported_types: tuple[type], name: str, priority: int,
                        solver: type[LinearSolver], defaults: dict[str, Any]):
        if name in self.defaults:
            raise ValueError(f'Solver with name {name} has already been registered.')
        self.solvers.append((priority, supported_types, name, solver))
        self.defaults[name] = defaults

    def get_solver(self, lhs: LinearOperator, context: str = '') -> LinearSolver:
        from nias.base.operators import LincombOperator

        if isinstance(lhs, LincombOperator):
            base_types = tuple(type(o) for o in lhs.operators)
        else:
            base_types = (type(lhs),)

        for priority, supported_types, name, solver in self.solvers:
            if all(isinstance(bt in supported_types) for bt in base_types):
                return solver(lhs, **self.defaults[name])

        raise InversionError(f'No solver known for base types {base_types}')

    def set_defaults(self, name: str, **values) -> None:
        self.defaults[name].update(values)

    def set_priority(self, name: str, priority: int) -> None:
        for i in range(len(self.solvers)):
            if self.solvers[i][2] == name:
                self.solvers[i][0] = priority
        self._sort_solvers()

    def _sort_solvers(self) -> None:
        self.solvers.sort(key=lambda x: x[0], reverse=True)

    def __str__(self):
        return ('Registered solvers\n------------------\n'
                + '\n'.join(f'{priority} [{", ".join(t.__name__ for t in types)}] {name} {self.defaults[name]}'
                            for priority, types, name, _ in self.solvers))


default_factory = DefaultLinearSolverFactory()
