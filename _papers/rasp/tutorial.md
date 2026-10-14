---
layout: paper-part
slug: rasp
part: tutorial
title: "Tutorial · Thinking Like Transformers"
---

## Overview

### Attention is all you need

\begin{align*}
\text{Attention}(x, W_Q, W_K, W_V) &= \text{Softmax}((W_Qx)^T W_Kx) (W_Vx) \\
\text{MultiHeadAttention}(x) &= \sum_i^N \text{Attention}(x, W_Q^i, W_K^i, W_V^i) W_O^i
\end{align*}

```python
def transfomer(x, depth):
    # x is a [L, D] matrix
    for _ in range(layers):
        mlp = MultiLayerPerceptron()
        x = map(mlp, x)

        mha = MultiHeadAttention()
        x = mha(x)
    return x
```

In practice transformers work very well! Attention really is all you need...
<!-- [@vaswani2023attention] -->

### Existing work

> Goal: Understand transformers. How do they work? Why?

Theory on transformers;

- Are they universal? Yes. [@yun2020transformers]
- But in practice they find some regular languages difficult.  [@bhattamishra2020ability]

Great. But we want a more mechanistic understanding of what they can do and how they do it, much like we have for RNNs (DFA extraction [@weiss_dfa]) and CNNs (feature and circuit extraction [@olah2020zoom]).

### RASP (restricted access sequence program)

<!-- can arbitrary combinations of select and aggregate be mapped onto transformers? -->

> [We propose] a computational model for the transformer-encoder in the form of a programming language [@weiss2021thinking]

- operations are applied to sequences
- operations;
    - input, indices, length
    - element-wise operators (correspond to the MLP layers)
    - select, aggregate (correspond to the MHA layers)

### Element wise operations

- math operations
    - addition
    - multiplication
    - power
- logical operations
    - and, or
    - less than, equal to
- other
    - max, min

Since MLPs are known to be universal function approximators, we can assume that we can represent any element wise operation as an MLP.
Thus we know that these RASP operations can be represented by the MLP layers of the transformer.

### Select, aggregate

Select

$$
\text{select}([0,1,2], [1,2,3], <) =
\begin{bmatrix}
T & F & F \\
T & T & F \\
T & T & T \\
\end{bmatrix}
$$

Aggregate

$$
\text{aggregate}(\begin{bmatrix}
T & F & F \\
T & T & F \\
T & T & T \\
\end{bmatrix}, [10, 20, 30]) =
[10, 15, 20]
$$

<!-- ### Selector width

$$
\text{selector width}(\begin{bmatrix}
T & F & F \\
T & T & F \\
T & T & T \\
\end{bmatrix}) = [1, 2, 3]
$$ -->

<!-- something they dont really touch on.
this is the power of linear algebra.
we can embed / represent logical operations into matrices / matmul.
aside: representation theory of groups!
 -->

### Examples

```python
def reverse(tokens):
    flip = select(indices, length-indices-1, ==)
    reverse = aggregate(flip, tokens)
    return reverse
```

```python
def sort(tokens):
    smaller = select(tokens, tokens, <=)
    target_pos = selector_width(smaller)
    sel_sort = select(target_pos, indices , ==)
    sort = aggregate(sel_sort, tokens)
    return sort
```

### Attention maps

![](/assets/images/rasp/weiss-attn-patterns-hist2.png)

### Results

![](/assets/images/rasp/weiss-results.png)

<!-- ### Insights

- the BOS acts as an extra memory cell
- they write a RASP program to sort a list of numbers (the algorithm is known to require $O(n log n)$ time). the transformer uses $n^2$.  -->

### Tracr

> We show how to “compile” human-readable programs [written in RASP] into standard decoder-
only transformer models. [@lindner2023tracr]

NOTE that the definition of a transformer has changed slightly here: all the layers are residual layers which read / write from the 'residual stream'.

### Example

```python
length = make_length() # `length` is not a primitive in our implementation.
opp_index = length - rasp.indices - 1
flip = rasp.Select(rasp.indices, opp_index, rasp.Comparison.EQ)
reverse = rasp.Aggregate(flip, rasp.tokens)

from tracr.compiler import compiling

bos = "BOS"
model = compiling.compile_rasp_to_model(
    reverse,
    vocab={1, 2, 3, 4, 5, 6, 7},
    max_seq_len=5,
    compiler_bos=bos)

out = model.apply([bos, 1, 2, 3, 1, 3])
out.decoded -> ['BOS', 3, 1, 3, 2, 1]
```

### Removed support for selector combination

The RASP language permits combining arbitrary selectors elementwise using boolean operators.

Let $S_1, S_2 \in \mathcal S$ be two selectors. And $m: \mathcal S \to [0, 1]^{d\times d}$ be a function that represents a selector as a matrix.


For every binary operation $\circ$ is there a matrix $M$ such that

\begin{align*}
S_1 \;\text{and}\; S_2 &= m(S_1) + m(S_2) \\
S_1 \;\text{or}\; S_2 &= m(S_1) ? m(S_2)
\end{align*}
