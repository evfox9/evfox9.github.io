---
title: 1-2. Neural Networks Basics
tags: AI Deep_Learning Coursera Deep_Learning_Specialization
---

## Logistic Regression as a Neural Network

### Binary Classification

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl1201.png)

**Binary classification** is labeling the inputs into two categories. For example, suppose we build a model that can
determines whether given image is a cat or not. When we input the image, it can have two labels: cat or non-cat. So this
is a binary classification model.

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl1202.png)

In this cat classifier model, the inputs are the images. Images can be represented as a $3$-dimensional matrices. To be
specific, it has a three $2$-dimensional matrices where each values represent the color of the corresponding pixel intensity
value. There are three images because color image has three channels: red, green and blue.

#### Notation

$(x, y)$: $x$ is an $n_x$-dimensional feature and $y$ is a label, which is $0$ or $1$

$m$: number of training examples

$m$ training examples: $(x^{(1)}, y^{(1)}),\ (x^{(2)}, y^{(2)}), \cdots , (x^{(m)}, y^{(m)})$

$X = \begin{bmatrix} \vdots & \vdots & \ & \vdots \\\ x^{(1)} & x^{(2)} & \cdots & x^{(m)} \\\ \vdots & \vdots & \ &
\vdots \end{bmatrix} \in \mathbb{R}^{n_x \times m}$

$Y = \begin{bmatrix} y^{(1)} & y^{(2)} & \cdots & y^{(m)} \end{bmatrix} \in \mathbb{R}^{1 \times m}$

### Logistic Regression

**Logistic Regression** is a learning algorithm that we can use for binary classification problems.

When we input $x$ to the model, it generates the estimated value of $y$, which we wrote as $\hat{y}$. Suppose
$w \in \mathbb{R}^{n_x}, b \in \mathbb{R}$ are parameters. Then, output $\hat{y} = w^T + b$ if it is a linear regression model.

But in binary classification, the output should be either $0$ or $1$, so the estimated output should be between $0$ and $1$,
which is the reason why we apply **sigmoid function** in logistic regression.

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl1203.jpg)

$$\sigma(z) = \frac{1}{1 + e^{-z}},\ z \in \mathbb{R},\ \sigma(z) \in (0,1)$$

We wrote $\hat{y} = \sigma(\theta^T x)$, where

$\theta = \begin{bmatrix} \theta_0 \\\ \theta_1 \\\ \theta_2 \\\ \vdots \\\ \theta_{n_x} \end{bmatrix}, \theta_0 = b$ and
$\theta_1 , \theta_2 , \cdots , \theta_{n_x} = w$

### Logistic Regression Cost Function

In logistic regression, our goal is to make ${\hat{y}}^{(i)}$ as closest to $y^{(i)}$ when $(x^{(1)}, y^{(1)}),\ (x^{(2)},
y^{(2)}), \cdots , (x^{(m)}, y^{(m)})$ is given. We can define **loss(error) function** to measure how well our algorithm is
doing. In logistic regression, we define loss function as $\mathcal{L}(\hat{y}, y) = -(y \log{\hat{y}} + (1 - y) \log{(1 - \hat{y})})$.

While loss function computes the error of a single training example, the **cost function** is the average of the loss functions
of the entire training set. We can write cost function as $J(w,b)=\frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})$.

### Gradient Descent

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl1204.png)

It seems natural that we want to minimize the cost function $J(w,b) = - \frac{1}{m} \sum_{i=1}^{m} y^{(i)} \log{\hat{y}^{(i)}} +
(1 - y^{(i)}) \log{(1 - \hat{y}^{(i)})} $. This cost function $J$ turns out to be a convex function, so no matter where we initialize
the function, we should get to roughly the same point. **Gradient descent** starts at the initial point and takes a step in the
  steepest downhill direction.

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl1205.png)

To apply gradient descent, we repeat updating the parameters by $w := w - \alpha \frac{\partial J(w)}{\partial w}$ where
\alpha is the learning rate.

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl1206.png)

From the graph above, we can easily show that $da = \frac{\partial \mathcal{L}(a,y)}{\partial a} = - \frac{y}{a} + \frac{1-y}{1-a}$
and $dz = \frac{\partial \mathcal{L}(a,y)}{\partial z} = \frac{\partial \mathcal{L}}{\partial a} \times \frac{\partial a}{\partial z}
= (- \frac{y}{a} + \frac{1-y}{1-a}) \times (a(1-a)) = a - y$.

We can expand this method to $m$ examples.

> $J = 0,\ d w_1 = 0,\ d w_2 = 0,\ db = 0$
>
> For $i = 1$ to $m$:
>
>   $z^{(i)} = w^T x^{i} + b$
>
>   $a^{(i)} = \sigma(z^{(i)})$
>
>   $J += - ( y^{(i)} \log{a^{(i)}} + (1 - y^{(i)}) \log{(1 - a^{(i)} ) } ) $
>
>   $d z^{(i)} = a^{(i)} - y^{(i)}$
>
>   $d w_1 += {x_1}^{(i)} d z^{(i)}$
>
>   $d w_2 += {x_2}^{(i)} d z^{(i)}$
>
>   $db += d z^{(i)}$
>
> $J /= m$
>
> $d w_1 /= m$

## Python and Vectorization

### Vectorization

**Vectorization** is basically the art of getting rid of explicit for-loops in the code. Avoiding the explicit for-loops makes
the algorithm run faster.

``` python
# Non-vectorized
z = 0
for i in range(n-x):
  z += w[i] * x[i]
  z += b

# Vectorized
z = np.dot(w,x) + b
```

### Broadcasting

In linear algebra, there are certain rules about matrix operations. For example, addition and subtraction is only allowed for
same size of matrices.

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl1207.png)

In `NumPy`, subject to certain constraints, it automatically expands the smaller array to have compatible shape during arithmetic
operations.

## Programming Assignment

[Logistic_Regression_with_a_Neural_Network_mindset](https://github.com/evfox9/Coursera/blob/master/Deep_Learning/Neural_Networks_and_Deep_Learning/Logistic_Regression_with_a_Neural_Network_mindset.ipynb)

---

## References

[Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning)
