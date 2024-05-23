---
file_format: mystnb
kernelspec:
  name: python3
---

# Mathematical Concepts

In this document, we will briefly discuss the mathematical concepts underlying NiAS' {mod}`~nias.interfaces`.

## Vector spaces

{{VectorSpaces}} in NiAS are either real or complex.
They can be finite or infinite dimensional.
Some are purely algebraic, others carry a norm or an inner product.

### {{VectorArrays}} and compatibility

Vectors are grouped in {{VectorArrays}}, ordered collections of vectors that are compatible in the
sense that linear combinations of these vectors can be formed in a unique way.
In other words, when two vectors $u_1,u_2$ are contained in vector spaces $X$ and $Y$, then any $X$-linear combination
of $u_1$ and $u_2$ has to agree with its corresponding $Y$-linear combination.
Similarly, we say that {{VectorArrays}} are compatible when linear combinations of arbitrary vectors from both arrays
can be formed.
In that case, it is allowed to {meth}`~nias.interfaces.VectorArray.append` vectors from one array to the other.

### {{VectorArrays}} and {{VectorSpaces}}

We say that a {{VectorArray}} $U$ is contained in a {{VectorSpace}} $X$ if all of its vectors are contained in that space.
{{VectorArrays}} can be contained in multiple {{VectorSpaces}}.
If two {{VectorArrays}} are contained in the same {{VectorSpace}}, then they are always compatible.
If $V$ is a compatible array that is not contained in $X$, then appending $V$ to $U$ will cause $U$ to be no
longer contained in $X$.

### Complex spaces and `scalar_type`

Complex {{VectorSpaces}} are always assumed to be the complexification of a real {{VectorSpace}}.
This means that for all complex {{VectorArrays}} there is a canonical conjugation operation defined.
In particular, we can extract the real and imaginary part of the vectors.

Each {{VectorArray}} and {{VectorSpace}} has an associated `scalar_type` which defines the real or complex floating-point
type that is used for scalar multiplication and internal storage.
Two arrays of the same kind which only differ in `scalar_type` are always compatible.
It is allowed to multiply a real array by a complex scalar to obtain a complexified array.
Not all {{VectorArray}} implementations will support complex numbers, though.
Except for complexification, in-place {{VectorArray}} methods never change the `scalar_type`.
A {{VectorArray}} $U$ is contained in a given {{VectorSpace}} $X$, irrespective of the `scalar_types` of $U$ and $X$,
as long as both are real or $X$ is a complex space.

### Anti-dual spaces

To each {{VectorSpace}} $X$, we can define its anti-dual space $X'$, consisting of anti-linear functionals
$f: X \to \mathbb{K}$, $\mathbb{K} \in \{\mathbb{R}, \mathbb{C}\}$.
Here, anti-linear means that for $u_1, u_2 \in X$, $\lambda_1, \lambda_2 \in \mathbb{K}$, we have

$$
f(\lambda_1 u_1 + \lambda_2 u_2) = \bar\lambda_1 f(u_1) + \bar\lambda_2 f(u_2).
$$

In particular, in the real case, $\mathbb{K} = \mathbb{C}$, the anti-dual space of $X$ is the same as the dual space of
$X$.
If $X$ is a normed space (see below), then we assume all functionals in $X'$ to be continuous.
When $X$ is finite dimensional, all anti-linear functionals on $X$ are automatically continuous.
In NiAS, each {{VectorSpace}} has its anti-dual accessible via the {attr}`~nias.interfaces.VectorSpace.antidual_space`
property.

#### The anti-bidual $X''$ of $X$

Considering the anti-bidual $X''$ of $X$, we have a canonical map
$$
\Phi_X: X \to X'' \qquad\text{with}\qquad \Phi_X[u](f) := \overline{f(u)} \quad \forall f \in X'.
$$
We easily see that $\Phi_X$ is linear:
$$
\begin{aligned}
\Phi_X[\lambda_1 u_1 + \lambda_2 u_2](f)
    &= \overline{f(\lambda_1 u_1 + \lambda_2 u_2)}\\
    &= \overline{\bar\lambda_1 f(u_1) + \bar\lambda_2 f(u_2)} \\
    &= \lambda_1 \overline{f(u_1)} + \lambda_2 \overline{f(u_2)} \\
    &= \lambda_1 \Phi_X[u_1](f) + \lambda_2 \Phi_X[u_2](f).
\end{aligned}
$$

For algebraic spaces $X$, it follows from Zorn's Lemma that $\Phi_X$ is always injective
(for each $x \in X$ there is an $f \in X'$ with $f(x) \neq 0$).
If $X$ is a normed space, the injectivity of $\Phi_X$ follows from the Hahn-Banach theorem.
We call $X$ reflexive, when $\Phi_X$ is an isomorphism.
All finite-dimensional spaces are reflexive, as well as all Hilbert spaces.
Further $L^p(X)$, is reflexive for $1 < p < \infty$.

```{important}
In NiAS, we assume all {{VectorSpaces}} $X$ to be reflexive.
Using $\Phi_X$, we identify $X''$ with $X$.
```

### The anti-dual pairing

Given a vector $u \in X$ and anti-linear form $f \in X'$, we define their anti-dual pairing $\langle u, f \rangle$
simply as the application of
$f$ on $u$:
$$
    \langle u, f \rangle := f(u).
$$
Thus, $\langle \cdot, \cdot \rangle$ defines a sesquilinear form on $X \times X'$.
It is anti-linear in the first variable $u$.
Identifying $X''$ with $X$, we also have
$$
    \langle f, u \rangle := \langle f, \Phi_X(u) \rangle = \Phi_X[u](f) = \overline{f(u)}
        = \overline{\langle u, f \rangle}.
$$

Note that the anti-dual pairing is independent of the (norm of) the spaces in which we consider the given vector/form.
Thus, in NiAS, it is available as a free function {func}`~nias.interfaces.dual_pairing` acting on arbitrary
{{VectorArrays}}.

```{note}
We consider *anti*-dual spaces in order to be able to identify sesquilinear forms with linear operators.
See below for further discussion.
```

## Vector spaces with bases

In NiAS, a {{VectorSpaceWithBasis}} is a finite-dimensional {{VectorSpace}} which carries a fixed chosen basis.
Given such a basis $b_1, \ldots, b_N$ of $X$, we assume it to be (practically) possible to obtain the coefficients
$\lambda_1,\ldots,\lambda_N$ of any vector $v \in X$, i.e.,
$$
v = \sum_{n=1}^N \lambda_n b_n.
$$
These coefficients can be obtained as a NumPy array (in Python) using the
{meth}`~nias.interfaces.VectorSpaceWithBasis.to_numpy` method of $X$.
Conversely, vectors can be constructed from basis coefficients using
{meth}`~nias.interfaces.VectorSpaceWithBasis.from_numpy`.

```{note}
Many spaces appearing in numerical analysis, e.g., finite-element spaces come with a basis, and vectors in these spaces
are actually represented as coefficient vectors w.r.t. this basis in memory.
Note, however, that there also many cases, where we do not have (efficient) access to basis coefficients, even if the
space is finite dimensional.
One example are subspaces of another space that statisfy some non-trivial linear constraint.
Also, there might be technical obstacles to obtain the basis coefficients.
Further, infinite-dimensional spaces do not have a (finite) basis.
Consequently, algorithms in NiAS try to avoid relying on basis representations whenever possible.
```

### Anti-dual basis

To each basis $b_1,\ldots,b_N$ of $X$, we can define an anti-dual basis $b_1^*,\ldots,b_N^*$ of $X'$ by requiring
$$
    b_i^*(b_j) := \delta_{i,j} =
        \begin{cases}
            1 & i = j \\
            0 & i \neq j,
        \end{cases}
$$
and extening the $b_i^*$ anti-linearly to all of $X$, yielding for $u = \sum_{j=1}^N \lambda_j b_j$ the formula
$$
    b_i^*(u) = b_i^*\left(\sum_{j=1}^N \lambda_j b_j\right) = \sum_{j=1}^N \bar\lambda_j b_i^*(b_j) = \bar\lambda_j.
$$
As every anti-linear form on $X$ is uniquely determined by its values on $b_1,\ldots,b_N$, it is easy to see that
$b_1^*,\ldots,b_N^*$, indeed, is a basis of $X'$.

```{important}
For every {{VectorSpaceWithBasis}}, we assume that its {attr}`~nias.interfaces.VectorSpace.antidual_space` is equipped
with the corresponding anti-dual basis.
```

What about the anti-dual basis of $X''$?
Since $\Phi_X[b_j](b_i^*) = \overline{b_i^*(b_j)} = \delta_{i,j}$, we see that we can identify $b_1,\ldots,b_N$ with the
anti-dual basis of $X''$.

### Anti-dual pairing in basis coefficients

Let $u = \sum_{j=1}^N \lambda_jb_j$ be a vector in $X$ and let $f = \sum_{i=1}^N \lambda_i^*b_i^*$ be an anti-linear
form on $X$.
Then we have:
$$
\begin{aligned}
    \langle u, f \rangle
    &= \left\langle \sum_{j=1}^N \lambda_jb_j, \sum_{i=1}^N \lambda_i^*b_i^*\right\rangle \\
    &= \left[\sum_{i=1}^N \lambda_i^*b_i^*\right]\left(\sum_{j=1}^N \lambda_jb_j\right)\\
    &= \sum_{i=1}^N \lambda_i^*b_i^*\left(\sum_{j=1}^N \lambda_jb_j\right)\\
    &= \sum_{i=1}^N\lambda_i^*\bar\lambda_i \\
    &= \underline{\bar\lambda}^T \cdot \underline{\lambda}^*,
\end{aligned}
$$
where $\underline{\lambda} = [\lambda_1,\ldots,\lambda_N]^T$, $\underline{\lambda}^* = [\lambda_1^*,\ldots,\lambda_N^*]^T$.
So in basis coefficients, the anti-dual pairing reduces to the Euclidian inner product of the coefficient vectors.
Identifying $u$ with $\Phi_X(u)$, we obtain
$$
    \langle f, u \rangle = \overline{\langle u, f\rangle}
        = \overline{\underline{\bar\lambda}^T \cdot \underline{\lambda}^*}
        = (\underline{\bar\lambda}^*)^T \cdot \underline{\lambda}
$$

## Normed spaces

Normed spaces are represented in NiAS by {{NormedSpace}} instances, which simply are {{VectorSpaces}} with an additional
{attr}`~nias.interfaces.NormedSpace.norm` attribute.
Since reflexive normed spaces are necessarily complete, all normed spaces in NiAS are Banach Spaces.

## Hilbert spaces

A {{SesquilinearForm}} $\varphi$ on $X \times Y$ is a mapping
$$
    \varphi: X \times Y \to \mathbb{K},
$$
that is anti-linear in the first variable and linear in the second variable.
In other words, we have
$$
    \varphi(\lambda_1 u_1 + \lambda_2 u_2, v) = \bar\lambda_1 \varphi(u_1, v) + \bar\lambda_2 \varphi(u_2, v)
    \qquad\text{and}\qquad
    \varphi(u, \mu_1v_1 + \mu_2v_2) = \mu_1 \varphi(u, v_1) + \mu_2 \varphi(u, v_2).
$$
For $\mathbb{K} = \mathbb{R}$, $\varphi$ simply is a bilinear form.

```{warning}
Choosing sesquilinear forms to be anti-linear in the first variable is merely a convention.
This convention is not universally agreed upon equally many authors assume sesquilinear forms to be anti-linear in the
second variable.
In particular, when interacting with complex numbers in other codes, make sure of you are aware which convention is
used.
```

A sesquilinear $\varphi$ form on $X \times X$ is called Hermitian if
$$
    \varphi(u_1, u_2) = \overline{\varphi(u_2, u_1)}.
$$
A Hermitian sesquilinear form is called positive definite or an inner product if
$$
    \varphi(u, u) > 0 \qquad\text{for all } u \neq 0.
$$
An inner product induces a norm via
$$
    \| u \| := \sqrt{\varphi(u, u)}.
$$

Now a {{HilbertSpace}} is a complete {{NormedSpace}} where the norm is induced by an inner product.
The inner product is accessible via the {attr}`~nias.interfaces.HilbertSpace.inner_product` attribute.

```{note}
As a single {{VectorArray}} can be contained in multiple {{NormedSpaces}} or {{HilbertSpaces}} at the same time, it is
not meaning full to speak of the norm of a {{VectorArray}} or the inner product between two {{VectorArrays}}.
Consequently, {{VectorArrays}} neither have methods for computing norms or inner products.
```

### Hilbert spaces with a basis

A {{HilbertSpaceWithBasis}} $X$ is a {{HilbertSpace}} with a fixed chosen basis $b_1,\ldots,b_N$.
This basis is *not* assumed to be an orthonormal basis.
If we denote by $(\cdot, \cdot)_X$ the inner product on $X$, we can define the matrix $M\in\mathbb{R}^{N\times N}$ of
$(\cdot, \cdot)_X$ by
$$
    M_{i,j} := (b_i, b_j)_X.
$$
Since $(\cdot, \cdot)_X$ is Hermitian, $M$ is a Hermitian matrix as well.
For $u = \sum_{i=1}^N \lambda_i b_i$, $v = \sum_{j=1}^N \mu_j b_j$, we can compute $(u, v)_X$ as
$$
\begin{aligned}
    (u, v)_X
    &= \left(\sum_{i=1}^N \lambda_i b_i, \sum_{j=1}^N \mu_j b_j \right) \\
    &= \sum_{i=1}^N \sum_{j=1}^N \bar\lambda_i \bar\mu_j M_{i,j} \\
    &= \underline{\bar\lambda}^T \cdot M \cdot \underline{\mu}.
\end{aligned}
$$

### Dual spaces of Hilbert spaces and the Riesz isomorphism

For a Hilbert Space $X$, we can associate with each $u \in X$ a corresponding anti-linear functional $\mathcal{R}_X(u)
\in X'$ given by
$$
    \mathcal{R}_X[u](v) := (v, u)_X.
$$
It is easily checked that $\mathcal{R}_X$ is a linear map, and the Riesz representation theorem states that this map is
actually an isomorphism.
Thus, every $f \in X'$ is of the form
$$
    f(v) = (v, u_f)_X.
$$
