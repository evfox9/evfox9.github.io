---
title: 1-4. Deep Neural Networks
tags: AI Deep_Learning Coursera Deep_Learning_Specialization
---

## Deep Neural Networks

### Deep L-layer Neural Network

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl1401.png)

We call a neural network **shallow** if it has relatively small number of hidden layers. On the other hand, **deep** neural
network has relatively large number of hidden layers.

### Matrix Dimensions

When implementing a deep neural network, one of the ways to check the correctness of the code is looking at the dimensions
of the matrices.

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl1402.png)

Picture above is 5-layer NN. When we implement forward propagation, first step will be $z^{[1]} = w^{[1]} x + b^{[1]}$.
Input layer has two features, so vector $x$ is going to be $(2,1)$ matrix, and $z^{[1]}$ is going to be $(3,1)$ vector.
Since $w^{[1]} x$ should have same dimension with $z^{[1]}$, $w^{[1]}$ should be $(3,2)$, because multiplying $(m,n)$ and
$(n,r)$ sized matrices return $(m,r)$ matrix.

With same method, we can figure out the dimension of the parameters.

$W^{[l]}: (n^{[l]}, n^{[l-1]})$

$b^{[l]}: (n^{[l]}, 1)$

$d W^{[l]}: (n^{[l]}, n^{[l-1]})$

$d b^{[l]}: (n^{[l]}, 1)$

$z^{[l]},\ a^{[l]}:\ (n^{[l]}, 1)$

$Z^{[l]},\ A^{[l]},\ d Z^{[l]},\ d A^{[l]}:\ (n^{[l]}, m)$

### Why deep representations?

Deep neural network can work better than shallow neural network.

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl1403.png)

Think of facial recognition system. In earlier layer, it detects simple functions like edges, and then composing them together
in the later layers of a neural network, which enables it to find more complex things.

### Forward, Backward Propagation

#### Forward Propagation for layer l
Input: $a^{[l-1]}$, Output: $a^{[l]}$, cache $z^{[l]}$

$z^{[l]} = w^{[l]} a^{[l-1]} + b^{[l]}$

$a^{[l]} = g^{[l]} (z^{[l]})$

#### Backward Propagation for layer l

Input: $d a^{[l]}$, Output: $d a^{[l-1]}, d W^{[l]}, d b^{[l]}$

$d z^{[l]} = d a^{[l]} \times g{[l]'} (z^{[l]}$

$d w^{[l]} = d z^{[l]} a^{[l-1]T}$

$d b^{[l]} = d z^{[l]}$

$d a^{[l-1]} = w^{[l]T} d z^{[l]}$

### Parameters, Hyperparameters

**Parameters**: $W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}, \cdots $

**Hyperparameters**: learning rate $\alpha$, # of iterations, # of hidden layer L, # of hidden units, choice of activation function,
momentum, minibatch size, regularization, etc.


## Programming Assignment

[Building_your_Deep_Neural_Network_Step_by_Step](https://github.com/evfox9/Coursera/blob/master/Deep_Learning/Neural_Networks_and_Deep_Learning/Building_your_Deep_Neural_Network_Step_by_Step.ipynb),
[Deep_Neural_Network_Application](https://github.com/evfox9/Coursera/blob/master/Deep_Learning/Neural_Networks_and_Deep_Learning/Deep_Neural_Network_Application.ipynb)

---

## References

[Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning)
