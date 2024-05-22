---
file_format: mystnb
kernelspec:
  name: python3
---

# Mathematical Concepts

In this document, we will briefly discuss the mathematical concepts underlying NiAS' {mod}`~nias.interfaces`.

## Vector spaces

{class}`VectorSpaces <nias.interfaces.VectorSpace>` in NiAS are either real or complex.
They can be finite or infinite dimensional.

Vectors are grouped in `VectorArrays`, ordered collections of vectors that are compatible in the
sense that linear combinations of these vectors can be formed in a unique way.
In other words, when two vectors $x_1,x_2$ are contained in vector spaces $X$ and $Y$, then any $X$-linear combination
of $x_1$ and $x_2$ has to agree with its corresponding $Y$-linear combination.
Similarly, we say that `VectorArrays` are compatible when linear combinations of arbitrary vectors from both arrays
can be formed.
In that case, it is allowed to {meth}`~nias.interfaces.VectorArray.append` vectors from one array to the other.

We say that a `VectorArray` `U` is contained in a `VectorSpace` `X` if all of its vectors are contained in that space.
`VectorArrays` can be contained in multiple `VectorSpaces`.
If two `VectorArrays` are contained in the same `VectorSpace`, then they are always compatible.
If `V` is a compatible array that is not contained in `X`, then appending `V` to `U` will cause `U` to be no
longer contained in `X`.

### Complex spaces and `scalar_type`

Complex `VectorSpaces` are always assumed to be the complexification of a real `VectorSpace`.
This means that for all complex `VectorArrays` there is a canonical conjugation operation defined.
In particular, we can extract the real and imaginary part of the vectors.

Each `VectorArray` and `VectorSpace` has an associated `scalar_type` which defines the real or complex floating-point
type that is used for scalar multiplication and internal storage.
Two arrays of the same kind which only differ in `scalar_type` are always compatible.
It is allowed to multiply a real array by a complex scalar to obtain a complexified array.
Not all `VectorArray` implementations will support complex numbers, though.
Except for complexification, in-place `VectorArray` methods never change the `scalar_type`.
A `VectorArray` `U` is contained in a given `VectorSpace` `X`, irrespective of the `scalar_types` of `U` and `V`,
as long as both are real or `X` is a complex space.

### Anti-dual spaces

To each `VectorSpace` $X$, we can define its anti-dual space $X'$, consisting of anti-linear functionals
$f: X \to \mathbb{K}$, $\mathbb{K} \in \{\mathbb{R}, \mathbb{C}\}$.
Here, anti-linear means that for $x_1, x_2 \in X$, $\lambda_1, \lambda_2 \in \mathbb{K}$, we have

$$
f(\lambda_1 x_1 + \lambda_2 x_2) = \bar\lambda_1 f(x_1) + \bar\lambda_2 f(x_2).
$$

In particular, in the real case, $\mathbb{K} = \mathbb{C}$, the anti-dual space of $X$ is the same as the dual space of
$X$.
If $X$ is a normed space (see below), then we assume all functionals in $X'$ to be continuous.
When $X$ is finite dimensional, all anti-linear functionals on $X$ are automatically continuous.

```{note}
We consider *anti*-dual spaces in order to be able to identify sesquilinear forms with linear operators.
See below for further discussion.
```

Considering the anti-bidual $X''$ of $X$, we have a canonical linear map

$$
\Phi_X: X \to X'' \qquad\text{with}\qquad \Phi_X(x)(f) := \overline{f(x)} \quad \forall f \in X'.
$$

For algebraic spaces $X$, it follows from Zorn's Lemma that $\Phi_X$ is always injective
(for each $x \in X$ there is an $f \in X'$ with $f(x) \neq 0$).
If $X$ is a normed space, the injectivity of $\Phi_X$ follows from the Hahn-Banach theorem.
We call $X$ reflexive, when $\Phi_X$ is an isomorphism.
All finite-dimensional spaces are reflexive, as well as all Hilbert spaces.
Further $L^p(X)$ is reflexive for $1 < p < \infty$.

```{important}
In NiAS, we assume all `VectorSpaces` $X$ to be reflexive.
```
