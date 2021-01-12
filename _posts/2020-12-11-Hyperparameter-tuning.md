---
title: 2-3. Hyperparameter Tuning, Batch Normalization and Programming Framework
tags: AI Deep_Learning Coursera Deep_Learning_Specialization
---

## Hyperparameter Tuning

### Tuning Process

When you train a data, there are many kinds of hyperparameters you should care about, such as $\alpha$, $\beta$, ($\beta_1,
\beta_2$ and $\epsilon$ if using Adam), num of layers, num of hidden units, learning rate decay, mini-batch size, etc. It turns
out that some of these hyperparameters are more important than others. In most cases, $\alpha$, the learning rate is the
most important hyperparameters to tune. Other than alpha, a few other hyperparameters you may tune next would be $\beta$,
number of hidden units and mini-batch size. Number of layers and learning rate decay would be next to those.

How would you select a set of values to explore? Suppose we have two hyperparameters.

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl2301.png)

In earlier generations of machine learning algorithms, it was common practice to sample the points in a grid like the left
picture, systematically explore these values and choose the best one. This was okay when the number of parameters was relatively
small. In deep learning, it is recommended to choose the points at random. Reason you do this is that it's difficult to know
in advance which hyperparameters are going to be the most important.

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl2302.png)

After finding the best hyperparameters by choosing randomly, you might zoom in to the smaller region of the hyperparameters
and then sample more densely within this space.

### Using an appropriate scale

Previously, we learned that it's necessary to choose your hyperparameters randomly. But it turns out that sampling at random
doesn't mean sampling uniformly at random in some cases.

For example, say that we're picking hyperparameter $\alpha$ that is between $0.0001$ and $1$. If we pick the sample by random
uniform distribution, number of samples between $0.0001$ and $0.001$ will be relatively small compared to number of samples
between $0.1$ and $1$. To solve the problem, you should pick the uniformly random sample $r$ from $[-4,0]$ and implement it
by $\alpha = 10^r$.

## Batch Normalization

### What is Batch Normalization?

One of the most important ideas of deep learning is an algorithm called **batch normalization**, created by two researchers
Sergey Loffe and Christian Szegedy. Batch normalization makes your hyperparameter search problem much easier and makes your
neural network much more robust.

You might remember that normalizing the input features can speed up learnings when training a model. In deep neural network,
normalizing $a^{[i]}$ or $z^{[i]}$ makes the training of $w^{[i+1]}$, $b^{[i+1]}$ faster. There are some debates in the deep learning
literature about whether you should normalize the value before the activation function, or whether you should normalize the
value after applying the activation function. In practice, normalizing $z^{[2]}$ is done more often.

When some intermediate values $z^{(1)}, \cdots , z^{(m)}$ in NN is given,

$\mu = \frac{1}{m} \sum z^{(i)}$

$\sigma^2 = \frac{1}{m} \sum (z^{i} - \mu)^2$

${z_{norm}}^{(i)} = \frac{z^{(i)} - \mu}{\sqrt{\sigma^2 + \epsilon}}$

$\tilde{z^{(i)}} = \gamma {z_{norm}}^{(i)} + \beta$ where $\gamma$, $\beta$ are learnable parameters of model

If $\gamma = \sqrt{\sigma^2 + \epsilon}$ and $\beta = \mu$, then $\tilde{z}^{(i)} = z^{(i)}$.

### Why Batch Normalization works?

Previously, we learned that normalizing the input features to mean $0$ and variance $1$ speed up learning and batch norm does similar thing.

Furthermore, batch norm makes weights, later or deeper than your network, more robust to changes to weights in earlier layers of the neural network.

Batch Norm as regularization

- Each mini-batch is scaled by the mean/variance computed on just that mini-batch.

- This adds some noise to the values $z^{[l]}$ within that minibatch. So similar to dropout, it adds some noise to each hidden layer's activations.

- This has a slight regularization effect.

## Multi-class Classification

### Softmax Regression

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl2303.png)

There's a generalization of logistic regression called **Softmax regression**. The less you make predictions where you're
trying to recognize one of multiple classes, rather than just recognize two classes.

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl2304.png)

Activation function: $t = e^{z^{[l]}}$ $a^{[l]} = \frac{e^{z^{[l]}}}{\sum_{i=1}^{4} t^i}$ ${a_i}^{[l]} = \frac{t_i}{\sum_{i=1}^{4}}$

Softmax examples

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl2305.png)
