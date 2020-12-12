---
title: 2-1. Practical aspects of Deep Learning
tags: AI Deep_Learning Coursera Deep_Learning_Specialization
---

## Setting up your Machine Learning Application

### Train / Dev / Test sets

In the previous era, people generally split the dataset into 70:30 (train:test) or 60:20:20 (train/dev/test). If the given dataset 
is small, these traditional ratios might work. If the dataset is big enough it is better to put more data to the train set.

One other trend of modern deep learning is that more people are training on mismatched train and test distributions. In this 
case, it is encouraged to make sure that the dev and test sets come from the same distribution.

### Bias / Variance

In the deep learning error, another trend is that there's been less discussion of what's called the bias-variance tradeoff. 

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl2101.png)

If we fit straight line to the give data, it won't be a good fit to the data, having a high bias. We say that this is `underfitting` 
the data. On the opposite end, maybe we can fit the data perfectly, but it isn't the great fit either. There's a classifier 
of high variance and this is `overfitting` the data. There might be some classifier in between with a medium level of complexity, 
it is much more reasonable fit to the data.

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl2102.png)

If train set error is 1% and dev set error is 11%, we can say that this model is overfitted. If the train set has error of 15% 
and dev set error of 16%, model is not even fitting the train set because humans achieve nearly to 0% error. If train set erorr 
is 15% and dev set error is 30%, it both have high bias and high variance. 

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl2103.png)

This is a basic recipe for machine learning. If your model has high bias, you should try to make your network bigger or train 
larger. If your model has high variance, you should consider regularization or putting more data.

## Regularizing your Neural Network

### Regularization

If you suspect your neural network is overfitting, one of the first things you should try is `regularization`. To add regularization 
to logistic regression, you add $\frac{\lambda}{2m} { {\Vert w \Vert}_2 }^2 $ to cost function $J(w,b) = \frac{1}{m} \sum_{i=1}^{m} 
\mathcal{L} (\hat{y}^{(i)}, y^{(i)})$, where $\lambda$ is a regularization parameter and ${ {\Vert w \Vert}_2 }^2 = \sum_{j=1}^{n_x} {w_j}^2 = w^T w$, 
which is a square of Euclidean norm (or L2 norm) of the prime to vector $w$. This is also called the `L2 regularization`, which 
is the most common type regularization. 

There is also `L1 regularization`, which adds $\frac{\lambda}{m} \sum_{i=1}^{n_x} |w| = 
\frac{\lambda}{m} { {\Vert w \Vert}_1 }$. If you use L1 regularization, the $w$ will end up being sparse, which means $w$ has 
a lot of zeros and it can help with compressing the model, because the set of parameters are zero so you need less memory to 
store the model.

$\lambda$ is called the `regularization parameter`, which is another hyperparameter that we might have to tune. 

In neural network, you should add $\frac{\lambda}{2m} \sum_{l=1}^{L} { \Vert w^{[l]} \Vert_F }^2 $ to cost function, where ${ \Vert w^{[l]} \Vert_F }^2 = 
\sum_{i=1}^{n^{[l]}} \sum_{j=1}^{[l-1]} {(w_{i,j})^{[l]}}^2 $, which is also called **Frobenius norm**. 

To apply regularization in gradient descent, we add $\frac{\lambda}{m} W^{[l]}$ to $d W^{[l]}$ in back propagation. If we apply 
this to gradient descent, whatever the matrix $w$ is, it's going to be smaller, and that's why we also call L2 regularization 
as **weight decay**.

Reason why regularization actually prevent overfitting is that if the regularization become large, the parameters $W$ will be small, 
so $z$ will also be relatively small, this makes activation function to be relatively linear, so the whole neural network will be 
computing something not too far from a big linear function which is relatively simple function compared to the complex non-linear 
function.

### Dropout Regularization

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl2104.png)

In `dropout regularization`, we set some probability of eliminating a node in neural networks each time you train. 

Dropout regularization works because we can't rely on any one feature, so have to spread out weights.

### Other Regularization methods

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl2105.png)

`Data augmentation` is a useful technique when you need more training data, when getting more data is expensive or impossible. 
For example, when we train image data, you can augment the dataset by flip, rotate, crop or distort the existing data.

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl2106.png)

`Early stopping` is the technique that you run gradient descent with plotting the training error or cost function with the 
dev set error and stop at some point that looks optimal.

## Setting up your Optimization Problem

### Normalization

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl2107.png)

`Normalization` is the technique that helps speed up your training. Normalizing has two steps. 

1) Subtract out or to zero out the mean. $\mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}$, $x := x - \mu$

2) Normalize variance. $\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)})^2 $, $x := \frac{x}{\sigma}$

In short, replace $x$ to $\frac{x-\mu}{\sigma}$.


![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl2108.png)

So why normalization can make training faster? Before we normalize the inputs, we should use smaller learning rate when 
we use gradient descent because it might need a lot of steps. On the other hand, after the normalization, it will have more 
spherical contours, then wherever you start gradient descent can pretty much go straight to the minimum, so we can take much 
larger steps.

### Vanishing / Exploding Gradients
One of the big problems in training neural network is `data vanishing` and `exploding gradients`. It means that when you train 
a very deep network, your derivatives or your slope can sometimes get either very big or very small, which makes the training 
difficult. 

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl2109.png)

For the simplicity, suppose we use a linear activation function. Then, output $y$ should be similar to $W^{[l]} W^{[l-1]} \cdots 
W^{[2]} W^{[1]}$. If we keep multiplying numbers, it will become exponentially big if the numbers are bigger than $1$, or close 
to zero if we keep multiplying numbers between $0$ and $1$. For long time, this was a hugh barrier to train deep neural 
networks.

### Weight Initialization
`Weight initialization` is about carefully choosing the random initialization for the neural network, and it is one of the 
ways that significantly reduce the vanishing / exploding gradient problem, although it's a partial solution.

One example of weight initialization is to set the variance of $W_i$ to be equal to $\frac{1}{n}$. When you use ReLU as a 
activation function, using $\frac{2}{n}$ for the variance of $W_i$ works better. If you are using a tanh activation function,
using $\sqrt{\frac{1}{n^{[l-1]}}}$ works better, and this is called **Xavier initialization**.

### Gradient Checking

When you implement back propagation, there is a test called `gradient checking`, that can help you make sure that your implementation 
of back propagation is correct. Take $W^{[1]}, b^{[1]}, \cdots , W^{[L]}, b^{[L]}$ and reshape into a big vector $\theta$ and 
take $d W^{[1]}, d b^{[1]}, \cdots , d W^{[L]}, d b^{[L]}$ and reshape into a big vector $d \theta$. 

For each $i$: $d \theta_{approx}[i] = \frac{J(\theta_1 , \theta_2 , \cdots, \theta_i + \epsilon , \cdots) - J(\theta_1 , 
\theta_2 , \cdots, \theta_i - \epsilon , \cdots)}{2 \epsilon} \approx d \theta [i] = \frac{\partial J}{\partial \theta_i}$.

Check $\frac{ \Vert d \theta_{approx} - d \theta \Vert_2 }{ \Vert d \theta_{approx} \Vert_2 + \Vert d \theta \Vert_2 }$. 
If this value is close to $10^{-7}$, it means that the derivative approximation is very likely to correct. If it's close to 
$10^{-5}$, it needs a careful look. If it's close to or bigger than $10^{-3}$, there might be a bug somewhere. 

There are some tips for implementing gradient check.

- Don't use in training (only to debug)

- If algorithm fails grad check, look at components to try to identify bug

- Remember regularization

- Doesn't work with dropout

- Run at random initialization; perhaps again after some training

## Programming Assignment

[Initialization](https://github.com/evfox9/Coursera/blob/master/Deep_Learning/Improving_Neural_Networks/Initialization.ipynb), 
[Regularization](https://github.com/evfox9/Coursera/blob/master/Deep_Learning/Improving_Neural_Networks/Regularization.ipynb), 
[Gradient Checking](https://github.com/evfox9/Coursera/blob/master/Deep_Learning/Improving_Neural_Networks/Gradient_Checking.ipynb)

---
## References

[Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network)
