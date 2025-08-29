#!/usr/bin/env python3
# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


# define substitutions here in rst syntax
substitutions = """
"""

def add_substition(qualname, singular=None, plural=True, kind='class'):
    global substitutions

    if singular is None:
        singular = qualname.split('.')[-1]
    substitutions += f'\n.. |{singular}| replace:: :{kind}:`~{qualname}`'

    if plural is True:
        plural = singular + 's'
    if plural:
        substitutions += f'\n.. |{plural}| replace:: :{kind}:`{plural} <{qualname}>`'

for name in [
        'EuclideanSpace',
        'HilbertSpace',
        'HilbertSpaceWithBasis',
        'HSLinearOperator',
        'LinearOperator',
        'NormedSpace',
        'Operator',
        'SesquilinearForm',
        'VectorArray',
        'VectorSpace',
]:
    add_substition(f'nias.interfaces.{name}')

add_substition('nias.interfaces.VectorSpaceWithBasis', plural='VectorSpacesWithBasis')


def convert_substitution(line):
    key, subst = line.split(' replace:: ')
    key = key.strip().replace('.. |', '').replace('|', '')
    assert ' ' not in key
    subst = subst.strip().replace(':', '{', 1).replace(':', '}', 2)
    return key, subst

myst_substitutions = dict(convert_substitution(line)
                          for line in substitutions.split('\n') if line != '')

if __name__ == '__main__':
    print(substitutions)
