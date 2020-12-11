---
title: 1-3. Shallow Neural Networks
tags: AI Deep_Learning Coursera Deep_Learning_Specialization
---

## Shallow Neural Network

### Neural Network Representation

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl1301.png)

We call the layers with input features $x_1, x_2, x_3$ as `input layer`. Layer in the middle are called 
`hidden layer`. Layer on the right with only one node is called `output layer`. We don't count the input layer, so the neural 
network above is a 2-layer NN.

### Computing a Neural Network's Output

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl1302.png)

$z^{[1]} = \begin{bmatrix} {w_1}^{[1]T} \\\ {w_2}^{[1]T} \\\ {w_3}^{[1]T} \\\ {w_4}^{[1]T} \end{bmatrix}
\begin{bmatrix} x_1 \\\ x_2 \\\ x_3 \end{bmatrix} + \begin{bmatrix} {b_1}^{[1]} \\\ {b_2}^{[1]} \\\ {b_3}^{[1]} \\\ {b_4}^{[1]} \end{bmatrix}
= \begin{bmatrix} {w_1}^{[1]T} x + {b_1}^{[1]} \\\ {w_2}^{[1]T} x + {b_2}^{[1]} \\\ {w_3}^{[1]T} x + {b_3}^{[1]} \\\ {w_4}^{[1]T} x + {b_4}^{[1]} \end{bmatrix}
= \begin{bmatrix} {z_1}^{[1]} \\\ {z_2}^{[1]} \\\ {z_3}^{[1]} \\\ {z_4}^{[1]} \end{bmatrix}$

In short, $z^{[1]} = W^{[1]} x + b^{[1]},\ a^{[1]} = \sigma(z^{[1]})$.

For $z^{[i]}$ which $i \geq 2$, $z^{[i]} = W^{[i]} a^{[i-1]}+ b^{[i]},\ a^{[i]} = \sigma(z^{[i]})$.



### Activation Functions

`Activation function` is the function that defines the output of that node. Here are some examples of activation functions.

#### Sigmoid

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl1303.png)

$$a = \frac{1}{1 + e^{-z}}$$

#### Hyperbolic tangent (tanh)

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl1304.png)

$$a = \tanh{z} = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

#### Rectified Linear Unit (ReLU)

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl1305.png)

$$a = \max (0,z)$$

#### Leaky ReLU

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl1306.png)

$$a = \max (0.01z,z)$$

* can replace 0.01 to other number

#### Why non-linear activation functions?

Functions above are all non-linear functions. Activation functions should be non-linear because no matter how much you compose
linear functions it will still be linear functions, so having many hidden layers won't have any meanings. Having non-linear 
function as activation function makes the model more expressive.

### Derivative of Activation functions

#### Sigmoid

$$g'(z) = g(z) (1 - g(z))$$

$$0 < g'(z) \leq \frac{1}{4}$$

#### Hyperbolic tangent (tanh)

$$g'(z) = 1 - {\tanh{z}}^2$$

$$0 < g'(z) < 1$$

#### Rectified Linear Unit (ReLU)

$$g'(z) = \begin{cases} 0 \ \text{if} \ z < 0 \\ 1 \ \text{if} \ z > 0 \end{cases} $$

#### Leaky ReLU

$$g'(z) = \begin{cases} 0.01 \ \text{if} \ z < 0 \\ 1 \ \text{if} \ z > 0 \end{cases} $$

### Gradient descent for Neural Networks

#### Forward Propagation

$z^{[1]} = W^{[1]} x + b^{[1]}$
$a^{[1]} = \sigma(z^{[1]})$.
$z^{[2]} = W^{[2]} a^{[1]}+ b^{[2]}$
$a^{[2]} = \sigma(z^{[2]})$.

#### Backward Propagation

$d z^{[2]} = A^{[2]} - Y$
$d w^{[2]} = \frac{1}{m} d z^{[2]} A^{[1]T}$
$d b^{[2]} = \frac{1}{m} \text{np.sum(}d z^{[2]} \text{, axis=1, keepdims=True)}$
$d z^{[1]} = w^{[2]T} d z^{[2]} \times g^{[1]'} z^{[1]}$
$d W^{[1]} = \frac{1}{m} d Z^{[1]} X^{[T]}$
$d b^{[1]} = \frac{1}{m} \text{np.sum(} d z^{[1]} \text{, axis=1, keepdims=True)}$

To keep the dimension of the matrix after sum operation, you should set the **keepdims** parameter to true. 

## Programming Assignment

[Github link](https://github.com/evfox9/Coursera/blob/master/Deep_Learning/Neural_Networks_and_Deep_Learning/Planar_data_classification_with_onehidden_layer.ipynb)

---
## References

[Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning)



<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>