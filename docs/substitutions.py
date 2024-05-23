# This file is part of the NiAS project (https://github.com/nias-project).
# Copyright NiAS developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


# define substitutions here in rst syntax
substitutions = """
.. |HilbertSpaces| replace:: :class:`HilberSpaces <nias.interfaces.HilbertSpace>`
.. |HilbertSpace| replace:: :class:`~nias.interfaces.HilbertSpace`
.. |HilbertSpacesWithBasis| replace:: :class:`HilbertSpacesWithBasis <nias.interfaces.HilbertSpaceWithBasis>`
.. |HilbertSpaceWithBasis| replace:: :class:`~nias.interfaces.HilbertSpaceWithBasis`
.. |NormedSpaces| replace:: :class:`NormedSpaces <nias.interfaces.NormedSpace>`
.. |NormedSpace| replace:: :class:`~nias.interfaces.NormedSpace`
.. |SesquilinearForms| replace:: :class:`SesquilinearForms <nias.interfaces.SesquilinearForm>`
.. |SesquilinearForm| replace:: :class:`~nias.interfaces.SesquilinearForm`
.. |VectorArrays| replace:: :class:`VectorArrays <nias.interfaces.VectorArray>`
.. |VectorArray| replace:: :class:`~nias.interfaces.VectorArray`
.. |VectorSpaceWithBasis| replace:: :class:`~nias.interfaces.VectorSpaceWithBasis`
.. |VectorSpaces| replace:: :class:`VectorSpaces <nias.interfaces.VectorSpace>`
.. |VectorSpace| replace:: :class:`~nias.interfaces.VectorSpace`
"""

def convert_substitution(line):
    key, subst = line.split(' replace:: ')
    key = key.strip().replace('.. |', '').replace('|', '')
    assert ' ' not in key
    subst = subst.strip().replace(':', '{', 1).replace(':', '}', 2)
    return key, subst

myst_substitutions = dict(convert_substitution(line)
                          for line in substitutions.split('\n') if line != '')
