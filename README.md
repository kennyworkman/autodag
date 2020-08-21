# AutoDAG: visualization of the AutoGrad computation graph

This is a fork of the pedagogical automatic differentiation library written by
Matt Johnson called [AutoDidact](https://github.com/mattjj/autodidact), which is
in turn a tutorial implementation based on [the full version of
Autograd](https://github.com/hips/autograd).

## Example Use

Run a `pip install -r requirements.txt` in your choice of virtual env.
ecosystem.

```
>>> import autograd.numpy as np
>>> from autograd import grad, visualize
>>> import matplotlib.pyplot as plt
>>> def tanh(x):
...     y = np.exp(-2.0 * x)
...     return (1.0 - y) / (1.0 + y)
...
>>> grad_tanh = grad(tanh)
>>> visualize_grad_tanh = visualize(tanh)
>>>
>>> grad_tanh(1.0)
0.419974341614026
>>>
>>> visualize_grad_tanh(1.0)
Forward pass...
Showing computed function values:


                     o 0.7615941559557649
                     |
                    / \
                   |   |
0.8646647167633873 o   o 1.1353352832366128
                   |__ |
                      \|
                       |
                       o 0.1353352832366127
                       |
                       |
                       |
                       o -2.0
                       |
                       |
                       |
                       o 1.0

Backward pass...
Showing Jacobian-vector products:


                      o 0.419974341614026
                      |
                      |
                      |
                      o -0.209987170807013
                      |
                      |
                      |
                      o -1.5516069851487515
                      |
                     / \
                    |   |
-0.6708099071708693 o   o 0.8807970779778823
                    |__ |
                       \|
                        |
                        o 1.0

0.419974341614026

>>> (tanh(1.0001) - tanh(0.9999)) / 0.0002  # Compare to finite differences
0.41997434264973155
```

## What is this?

This is a simple implementation of the computational building blocks driving
optimization in the most popular artificial neural network libraries today
(pytorch, tensorflow).

You can see simple directed graph visualizations of:

  * computed function values from the initial forward pass
  * computed Jacobian-Vector products (JVPs) that compose the "backpropogation"
  to efficiently derive the composite function gradient

The principles of automatic differentiation in a functional programming context
are derived from the ideas of [Lambda Calculus](https://en.wikipedia.org/wiki/Lambda_calculus).

We are able to imitate the functional abstraction native to closure-native
languages like Lisp in Python by "wrapping" functional primitives as they
execute and constructing a tree-like structure from the stack trace.

Jacobian-Vector products are then simply derived from a mapping of existing
primitives and evaluated as the linear transformations (functions) that they are
rather than computing matrix multiplication.

Benefits of constructing a computation graph "on the fly" are twofold:

  * Familiar language mechanics like for-loops and branches can be used without
  some weird domain language. One less thing to memorize.
  * The cost of computing a gradient is fractionally greater than executing the
  function itself.

Check out Dougal Maclaurin's PhD [thesis](https://dougalmaclaurin.com/phd-thesis.pdf) for a more detailed and
mathematically rigorous breakdown of this approach.

## CLI Graph

The ASCII visualization was handled by
[graphterm](https://github.com/kisaacs/graphterm), which was developed to
visualize software dependencies in supercomputing. (Pretty interesting, read
[here](https://arxiv.org/abs/1908.07544)).

Please refer to the `requirements.txt` in this repo for my fork of this library
so it can be built and imported by pip.

### Autograd / Autodidact

Autograd was written by [Dougal Maclaurin](https://dougalmaclaurin.com),
[David Duvenaud](https://www.cs.toronto.edu/~duvenaud/)
and [Matt Johnson](http://people.csail.mit.edu/mattjj/).
See [the main page](https://github.com/hips/autograd) for more information.
