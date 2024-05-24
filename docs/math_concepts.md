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

### Vector arrays and compatibility

Vectors are grouped in {{VectorArrays}}, ordered collections of vectors that are compatible in the
sense that linear combinations of these vectors can be formed in a unique way.
In other words, when two vectors $u_1,u_2$ are contained in vector spaces $X$ and $Y$, then any $X$-linear combination
of $u_1$ and $u_2$ has to agree with its corresponding $Y$-linear combination.
Similarly, we say that {{VectorArrays}} are compatible when linear combinations of arbitrary vectors from both arrays
can be formed.
In that case, it is allowed to {meth}`~nias.interfaces.VectorArray.append` vectors from one array to the other.

### Vector arrays and vector spaces

We say that a {{VectorArray}} $U$ is contained in a {{VectorSpace}} $X$ if all of its vectors are contained in that space.
{{VectorArrays}} can be contained in multiple {{VectorSpaces}}.
If two {{VectorArrays}} are contained in the same {{VectorSpace}}, then they are always compatible.
If $V$ is a compatible array that is not contained in $X$, then appending $V$ to $U$ will cause $U$ to be no
longer contained in $X$.

### Complex spaces and scalar_type

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

To each {{VectorSpace}} $X$, we can define its anti-dual space $X'$, consisting of anti-linear forms
$f: X \to \mathbb{K}$, $\mathbb{K} \in \{\mathbb{R}, \mathbb{C}\}$.
Here, anti-linear means that for $u_1, u_2 \in X$, $\lambda_1, \lambda_2 \in \mathbb{K}$, we have

$$
f(\lambda_1 u_1 + \lambda_2 u_2) = \bar\lambda_1 f(u_1) + \bar\lambda_2 f(u_2).
$$

In particular, in the real case, $\mathbb{K} = \mathbb{C}$, the anti-dual space of $X$ is the same as the dual space of
$X$.
If $X$ is a normed space (see below), then we assume all anti-linear forms in $X'$ to be continuous.
When $X$ is finite dimensional, all anti-linear forms on $X$ are automatically continuous.
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
and extending the $b_i^*$ anti-linearly to all of $X$, yielding for $u = \sum_{j=1}^N \lambda_j b_j$ the formula
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

### Anti-dual pairing in coordinates

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
    &= \underline{\lambda}^H \cdot \underline{\lambda}^*,
\end{aligned}
$$
where $\underline{\lambda} = [\lambda_1,\ldots,\lambda_N]^T$, $\underline{\lambda}^* = [\lambda_1^*,\ldots,\lambda_N^*]^T$
and $\underline{\lambda}^H := \overline{(\underline{\lambda}^T)}$.
So in basis coefficients, the anti-dual pairing reduces to the Euclidean inner product of the coefficient vectors.
Identifying $u$ with $\Phi_X(u)$, we obtain
$$
    \langle f, u \rangle = \overline{\langle u, f\rangle}
        = \overline{\underline{\lambda}^H \cdot \underline{\lambda}^*}
        = (\underline{\lambda}^*)^H \cdot \underline{\lambda}
$$

## Normed spaces

Normed spaces are represented in NiAS by {{NormedSpace}} instances, which simply are {{VectorSpaces}} with an additional
{attr}`~nias.interfaces.NormedSpace.norm` attribute.
Since reflexive normed spaces are necessarily complete, all normed spaces in NiAS are Banach Spaces.

### Norm of the anti-dual space

When $X$ is a normed space with norm $\|\cdot\|_X$, then $X'$ carries a natural dual norm, which for $f \in X'$ is given
by
$$
    \|f\|_{X'} := \sup_{0 \neq u \in X} \frac{|f(u)|}{\|u\|_X}.
$$
If $X$ is a {{NormedSpace}}, we assume `X.antidual_space` to also be a {{NormedSpace}} equipped with this dual norm.

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
    &= \underline{\lambda}^H \cdot M \cdot \underline{\mu}.
\end{aligned}
$$ (eq:inner_product_matrix)

### Dual spaces of Hilbert spaces and the Riesz isomorphism

For a Hilbert Space $X$, we can associate with each $u \in X$ a corresponding anti-linear forms $\mathcal{R}_X(u)
\in X'$ given by

$$
    \mathcal{R}_X[u](v) := (v, u)_X.
$$ (eq:def_riesz)

It is easily checked that $\mathcal{R}_X$ is a linear map, and the Riesz representation theorem states that this map is
actually an isomorphism.
Thus, every $f \in X'$ is of the form
$$
    f(v) = (v, u_f)_X.
$$
Furthermore, the Riesz isomorphism $\mathcal{R}_X$ is isometric, meaning that $\|\mathcal{R}_X(u)\|_{X'} = \|u\|_X$.
As $\mathcal{R}_X^{-1}$ is linear, we can define an inner product on $X'$ given by
$$
    (f, g)_{X'} := (\mathcal{R}_X^{-1}(f), \mathcal{R}_X^{-1}(g))_X.
$$
This inner product induces the dual norm on $X'$ since $\mathcal{R}_X$ is isometric.
Further, we get
$$
    \langle u, \mathcal{R}_X(u) \rangle = \mathcal{R}_X(u)(u) = (u, u)_X = \|u\|_X^2.
$$
Similarly
$$
    \langle \mathcal{R}_X^{-1}(f), f \rangle
    = f(\mathcal{R}_X^{-1}(f))
    = (\mathcal{R}_X^{-1}(f), \mathcal{R}_X^{-1}(f))_X
    = (f, f)_{X'}
    = \|f\|_{X'}^2.
$$

For a NiAS {{HilbertSpace}} the Riesz isomorphism $\mathcal{R}_X$ is available via the
{meth}`~nias.interfaces.HilbertSpace.riesz` method.

### The anti-bidual space of a Hilbert space

How does the Hilbert space geometry of $X''$ look like?
Let some $\mathcal{R}_X(u) \in X'$ be given.
Then $\mathcal{R}_{X'}(\mathcal{R}_{X}(u))$ is given by
$$
\begin{aligned}
   \mathcal{R}*{X'}[\mathcal{R}_{X}(u)](f)
    &= (f, \mathcal{R}*{X}(u))*{X'}\\
    &= (\mathcal{R}*{X}^{-1}(f), u)*{X}\\
    &= \overline{(u, \mathcal{R}*{X}^{-1}(f))_{X}}\\
    &= \overline{f(u)} \\
    &= \Phi_X[u](f),
\end{aligned}
$$
so
$$
    \mathcal{R}_{X'} = \Phi_X \circ \mathcal{R}_{X}^{-1}.
$$
It follows
$$
\begin{aligned}
    (\Phi_X(u), \Phi_X(v))*{X''}
    &:= (\mathcal{R}*{X'}^{-1}(\Phi_X(u)), \mathcal{R}*{X'}^{-1}(\Phi_X(v))*{X'} \\
    &= (\mathcal{R}*{X}(\Phi_X^{-1}(\Phi_X(u))), \mathcal{R}*{X}(\Phi_X^{-1}(\Phi_X(v)))*{X'} \\
    &= (\mathcal{R}*{X}(u), \mathcal{R}*{X}(v))*{X'}  \\
    &= (u, v)_X.
\end{aligned}
$$
So, identifying $X''$ with $X$ via $\Phi_X$, also $(\cdot, \cdot)_{X''}$ agrees with $(\cdot, \cdot)_X$,
and we can say that $X'' = X$ as Hilbert spaces.

### Riesz isomorphism and dual norm in coordinates

Let $b_1,\ldots,b_N$ a basis of a Hilbert space $X$ and let $M$ be the matrix of $(\cdot, \cdot)_X$ as defined
[above](#hilbert-spaces-with-a-basis).
Let $u = \sum_{i = 1}^N \lambda_i b_i$ with coefficient vector $\underline{\lambda} = [\lambda_1, \ldots, \lambda_N]^T$.
Then consider
$$
    \underline{\lambda}^* := M \cdot \underline{\lambda} \qquad\text{and}\qquad
    f := \sum_{j=1}^N \lambda^*_j b_j^* \in X',
$$
where $b_1,\ldots,b_N^*$ again denotes the dual basis.
For another arbitrary $v = \sum_{j=1}^n \mu_j b_j$, $\underline{\mu} = [\mu_1, \ldots, \mu_N]^T$ we have
$$
\begin{aligned}
    f(v) &= <v, f> \\
         &= \underline{\mu}^H \cdot \underline{\lambda}^* \\
         &= \underline{\mu}^H \cdot M \cdot \underline{\lambda} \\
         &= (v, u)_X.
\end{aligned}
$$
Hence, $f = \mathcal{R}_X(u)$, and we have shown that in coordinates, computing $\mathcal{R}_X(u)$ boils down to
multiplying the coefficient vector of $u$ with $M$.
Conversely, $\mathcal{R}_X^{-1}$ is given by multiplication with $M^{-1}$.
It follows, that the $X'$-inner product of $f := \sum_{j=1}^N \lambda^*_j b_j^*$ and $g := \sum_{j=1}^N \mu^*_j b_j^*$
is given by
$$
\begin{aligned}
    (f, g)_{X'}
    &= (\mathcal{R}_X^{-1}(f), \mathcal{R}_X^{-1}(g))_X\\
    &= (M^{-1}\cdot \underline{\lambda}^*)^H \cdot M \cdot (M^{-1}\cdot\underline{\mu}^*)\\
    &= (\underline{\lambda}^*)^H \cdot M^{-1} \cdot \underline{\mu}^*.
\end{aligned}
$$

## Euclidean spaces

We call a {{HilbertSpaceWithBasis}} $X$ an {{EuclideanSpace}} when its basis $b_1,\ldots,b_N$ is an orthonormal basis,
i.e.
$$
    (b_i, b_j)_X = \delta_{i,j}.
$$
In that case, the inner-product matrix $M$ is just the identity matrix.
It follows that the matrix of the Riesz isomorphism $\mathcal{R}_X$ and the matrix of the $X'$-inner product is the
identity as well.
Hence, if $u = \sum_{i=1}^N \lambda_ib_i$ and $\mathcal{R}_X(u) = \sum_{i=1}^n \lambda^*_ib_i^*$, then
$$
    \underline{\lambda} = \underline{\lambda}^*,
$$
and for $v = \sum_{i=1}^{N} \mu_ib_i$, we have
$$
    \langle v, \mathcal{R}_X(u) \rangle
    = \underline{\mu}^H \cdot \underline{\lambda}
    = (v, u)_X.
$$
Hence, in coordinates, the $X$-inner product reduces to the Euclidean inner product, and it does not matter whether we
interpret a coefficient vector $\underline{\lambda}$ as an element of $X$ or $X'$.
Consequently we identify the {attr}`~nias.interfaces.VectorSpace.antidual_space` of an {{EuclideanSpace}} with itself.

## Operators

Every mapping between two {{VectorSpaces}} is called an {{Operator}} in NiAS.
The mapping can be applied to any {{VectorArray}} in the operator's {attr}`~nias.interfaces.Operator.source_space`
using the {meth}`~nias.interfaces.Operator.apply` method, yielding a {{VectorArray}} in the operator's
{attr}`~nias.interfaces.Operator.range_space`.
In particular, {attr}`~nias.interfaces.Operator.source_space` and {attr}`~nias.interfaces.Operator.range_space` are
allowed to differ.
If both spaces are {{NormedSpaces}}, we assume the operator to be continuous.
There is no notion of a matrix in NiAS.
Matrices are represented by the linear {{Operator}} given by left-multiplication with the given matrix.

## Linear operators

A {{LinearOperator}} $T: X \to Y$ is simply an {{Operator}} which is assumed to be linear in its argument, i.e.,
$$
    T(\lambda_1 u_1 + \lambda_2 u_2) = \lambda_1 T(u_1) + \lambda_2 T(u_2).
$$

### Linear operators in coordinates

Let $b_1,\ldots,b_N$, $c_1,\ldots,c_M$ be bases of $X$ and $Y$.
Then to each linear operator $T: X \to Y$ we can assign a matrix $A \in \mathbb{M\times N}$ where the $j$-column
$A_{:,j}$ of $A$ contains the coefficients of $T(b_j)$ w.r.t. $c_1,\ldots,c_M$, i.e.,
$$
    T(b_j) = \sum_{i_1}^{M} A_{i,j}c_i.
$$
By linearity of $T$ it follows that for any $u = \sum_{i=1}^{N} \lambda_ib_i$ we have that $T(u) =:
\sum_{i=1}^{M}\mu_ic_i$ has the coefficient vector
$$
    \underline{\mu} = A \cdot \underline{\lambda}.
$$

## The transpose of a linear operator

For any (continuous) linear operator $T: X \to Y$, there is a (continuous) linear transpose operator $T^t: Y' \to X'$
given by:
$$
    T^t[g](u) := g(T(u)).
$$
Note that $T^t[g]$, indeed, is an anti-linear form since
$$
\begin{aligned}
    T^t[g](\lambda_1 u_1 + \lambda_2 u_2)
    &= g(T(\lambda_1 u_1 + \lambda_2 u_2)) \\
    &= g(\lambda_1 T(u_1) + \lambda_2 T(u_2)) \\
    &= \bar\lambda_1 g(T(u_1)) + \bar\lambda_2 g(T(u_2)) \\
    &= \bar\lambda_1 T^t[g](u_1) + \bar\lambda_2 T^t[g](u_2).
\end{aligned}
$$
NiAS {{LinearOperators}} have a {meth}`~nias.interfaces.LinearOperator.apply_transpose` method to apply the transpose of
the given operator.
Note that the definition of the transpose operator is purely algebraic and independent of any potential norms or inner
products on $X$ or $Y$.

### Transpose operator in coordinates

Let $A \in \mathbb{R}^{M\times N}$ be the matrix of a linear operator $T: X \to Y$, and let $b_1,\ldots,b_N$,
$c_1,\ldots,c_M$ be bases of $X$ and $Y$.
$c_1,\ldots,c_M$ be bases of $X$ and $Y$.
Let $B \in \mathbb{R}^{N \times M}$ bet the matrix of $T^t$ w.r.t. the dual bases $c_1^*,\ldots,c_N^*$,
$b_1^*,\ldots,b_M^*$ of $Y'$ and $X'$.
Then we have
$$
    B_{i,j} = T[c_j^*](b_i) = c_j^*(T(b_i)) = \bar A_{j,i},
$$ (eq:transpose_coordinates)
so the matrix of $T^t$ is given by the conjugate transpose $A^H$ of $A$.

```{note}
The matrix of $T^t$ is $A^H$ and not $A^T$ since we consider anti-dual spaces.
If we define an analogue transpose operator between the linear dual spaces of $Y$ and $X$,
we will obtain $A^T$ as the matrix of this operator.
```

## The adjoint of a linear Hilbert space operator

For {class}`linear Hilbert space operators <nias.interfaces.HSLinearOperator>` $T: X \to Y$ between Hilbert spaces $X$
and $Y$, the adjoint operator $T^*: Y \to X$ is determined by the relation
$$
    (T(u), v)_Y = (u, T^*(v))_X.
$$
Note that
$$
    (T(\cdot), v)_Y = \mathcal{R}_Y[v](T(\cdot)) = T^t(\mathcal{R}_Y(v)),
$$
which is a continuous anti-linear functional on $X$.
Hence, by the Riesz representation theorem, $T^*(v)$ is given by
$$
    T^*(v) = \mathcal{R}_X^{-1}(T^t(\mathcal{R}_Y(v))),
$$
so

$$
    T^* = \mathcal{R}_X^{-1} \circ T^t \circ \mathcal{R}_Y.
$$ (eq:relation_adjoint_transpose)

### Adjoint operator in coordinates

Let $M_X \in \mathbb{R}^{N\times N}$, $M_Y \in \mathbb{R}^{M\times M}$ be the matrices of $(\cdot,\cdot)_X$ and $(\cdot,
\cdot)_Y$, and let $A \in \mathbb{R}^{M\times N}$ be the matrix of $T$.
Then by {eq}`eq:inner_product_matrix`, {eq}`eq:def_riesz` and {eq}`eq:relation_adjoint_transpose`, we get that the
matrix of $T^*$ is given by

$$
   M_X^{-1} \cdot A^H \cdot M_Y.
$$

## Sesquilinear forms and operators

Given a {{SesquilinearForm}} $\varphi: X \times Y \to \mathbb{K}$, note that for each $v \in Y$, $\varphi(\cdot, v)$ is
an anti-linear form on $X$.
Thus,
$$
    T: Y \to X',\quad T(v) := \varphi(\cdot, v)
$$
defines an {{Operator}}, which is easily seen to be linear.

Conversely, if $T: Y \to X'$ is a {{LinearOperator}}, we can associate a {{SesquilinearForm}}
$$
\varphi: X' \times Y \to \mathbb{K}, \quad \varphi(u, v):= T(v)(u) = \langle u, T(v) \rangle.
$$
This situation is relevant, for instance, in finite-element methods, were a weak formulation
$$
    \varphi(v, u) = f(v) \qquad\text{for all } v\in X,
$$
is interpreted as
$$
    T(u) = f.
$$

Note that $T$ and $\varphi$ always have the same matrix representations when $X'$ is equipped with the dual basis to the
basis chosen on $X$.

Given $T: Y \to X'$, the corresponding {{SesquilinearForm}} $\varphi$ can be obtained via the
{meth}`~nias.interfaces.LinearOperator.as_sesquilinear_form` interface method of {{LinearOperator}}.

### Sesquilinear forms and operators on Hilbert spaces

Given a sesquilinear form $\varphi: X \times Y \to \mathbb{K}$, where $X$ is a Hilbert space, we can make use of the
Riesz isomorphism to associate a linear operator mapping to $Y$ instead of $Y'$ by defining
$$
    T: Y \to X,\quad T(v) := \mathcal{R}_X^{-1}(\varphi(\cdot, v)).
$$

For the converse direction, given a linear operator $T: Y \to X$, we obtain a sesquilinear form on $X \times Y$, by
defining
$$
    \varphi(v, u) := (v, T(u))_X.
$$

Note that if $A$ is the matrix of $T$, then $\varphi$ has the matrix representation
$$
    M \cdot A,
$$
where $M$ is the matrix of the $X$-inner product.
Thus, the matrix representations of $T$ and $\varphi$ do *not* agree, unless $X$ is a
[Euclidean space](#euclidean-spaces).
