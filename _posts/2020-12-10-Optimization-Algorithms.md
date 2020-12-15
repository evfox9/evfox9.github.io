---
title: 2-2. Optimization Algorithms
tags: AI Deep_Learning Coursera Deep_Learning_Specialization
---

## Optimization Algorithms

### Mini-batch Gradient Descent

Vectorization allows you to efficiently compute on m examples. It turns out that you can get a faster algorithm if you let 
gradient descent start to make some progress even before you finish processing your entire training set. Specifically, you 
split up your training set into smaller training sets, and these sets are called **mini-batches**.

Then when we use mini-batch and how do we choose the size of mini-batch? If training set is small ($m \leq 2000$), use batch 
gradient descent. Otherwise, typical mini-batch size will be somewhere between $64$ and $512$, and because of the way computer 
memory lay out and access, sometimes your code runs faster if your mini-batch size is a power of $2$. Also, make sure that all 
of your mini-batch $X^{ \{ t \} }, Y^{ \{ t \} }$ fits in CPU/GPU memory, or the performance might falls off a cliff. In practice, 
mini-batch size is another hyperparameter that you might do a quick search over to try to figure out which one is more sufficient 
of reducing the cost function $J$. So just try several different value and choose what looks optimal.

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl2201.png)

### Exponentially Weighted Averages

There are some optimization algorithms that runs faster than gradient descent. These algorithms use **exponentially weighted 
averages**. 

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl2202.png)

For example, imagine that we have the daily temperature data of London and you want to compute the trends. First, let's initialize 
$v_0 = 0$ and then, on everyday we're going to average it with a weight of 0.9 times whatever appears as value, plus 0.1 times 
that day temperature, so $v_1 = 0.9 v_0 + 0.1 \theta_1 $ where $\theta_1$ is the temperature of 1st day. If we plot 
$v$ in red line, you get a moving average of exponentially weighted average of the daily temperature.

Let's look at the equation. 

$$v_t = \beta v_{t-1} + (1 - \beta) \theta_t$$

You can think of $v_t$ as approximately averaging over $\frac{1}{1 - \beta}$ days' temperature and $\beta$ is a hyperparameter. 
If we set $\beta$ larger, it is giving more weight to the previous value, so whether the temperature changes, exponentially 
weighted average adapts more slowly. If we set $\beta$ smaller, it only averages over small number of days, so it will be more 
noisy and susceptible to outliers, but this adapts much more quickly to temperature changes.

There is a technique called bias correction that can make you compute the averages more accurately. Suppose $\beta = 0.98$.
Then, $v_1 = 0.98 v_0 + 0.02 \theta_1$ and since we initialize the $v_0$ to $0$, $v_1 = 0.02 \theta_1$. This might be problematic 
because $0.02 \theta_1 $ is quite a small value, and it also affects the later average since $v_2 = 0.98 v_1 + 0.02 \theta_2$. 
There is a way to fix this kind of problem which occurs in initial phase, which is taking $\frac{v_t}{1 - \beta^t}$ instead of 
$v_t$. In initial phase, where $t$ is small, $1 - \beta^t$ will be relatively small, which makes $v_t$ larger. Later on, $t$ 
will become larger and $1 - \beta^t$ will also become larger, which won't affect $v_t$ much. 

### Gradient Descent with Momentum

There's an algorithm called **momentum**, that almost always works faster than the standard gradient descent algorithm. Basic 
idea is to compute an exponentially weighted average of your gradients and then use that gradient to update your weights instead.

On iteration t: compute $dW$, $db$ on the current mini-batch 

$v_{dW} = \beta v_{dW} + (1 - \beta) dW$

$v_{db} = \beta v_{db} + (1 - \beta) db$

$W = W - \alpha v_{dW},\ b = b - \alpha v_{db}$

Hyperparameters: $\alpha$, $\beta$

$\beta = 0.9$ is the most common value

### RMSprop

**RMSprop**, stands for root mean square prop, is an algorithm that can also speed up gradient descent. 

On iteration t: compute $dW$, $db$ on the current mini-batch 

$S_{dW} = \beta S_{dW} + (1 - \beta) dW^2$

$S_{db} = \beta S_{db} + (1 - \beta) db^2$

$W = W - \alpha \frac{dW}{ \sqrt{ S_{dW} } + \epsilon } , \ b = b - \alpha \frac{db}{ \sqrt{ S_{db} } + \epsilon }$

### Adam

**Adam**, stands for adaptive moment estimation, is one of the stood up algorithm that has been shown to work well across 
a wide range of deep learning architectures. 

On iteration t: compute $dW$, $db$ using current mini-batch 

$V_{dW} = \beta_1 V_{dW} + (1 - \beta_1) dW,\ V_{db} = \beta_1 V_{db} + (1 - \beta_1) db \ \leftarrow \text{"momentum"} \ \beta_1$

$S_{dW} = \beta_2 S_{dW} + (1 - \beta_2) dW^2 ,\ S_{db} = \beta_2 S_{db} + (1 - \beta_2) db^2 \ \leftarrow \text{"RMSprop"} \ \beta_2$

$V_{dW}^{corrected} = \frac{V_{dW}}{(1 - \beta_{1}^{t})},\ V_{db}^{corrected} = \frac{V_{db}}{(1 - \beta_{1}^{t})}$

$S_{dW}^{corrected} = \frac{S_{dW}}{(1 - \beta_{2}^{t})},\ S_{db}^{corrected} = \frac{S_{db}}{(1 - \beta_{2}^{t})}$

$W := W - \alpha \frac{ V_{dW}^{corrected} }{ \sqrt{ S_{dW}^{corrected} } + \epsilon } $
$b := b - \alpha \frac{ V_{db}^{corrected} }{ \sqrt{ S_{db}^{corrected} } + \epsilon } $

Hyperparameters choice: 

$\alpha$: needs to be tune

$\beta_1$ : $0.9$ is the default choice ($dW$)

$\beta_2$ : $0.9$ is recommended ($dW^2$)

$\epsilon$ : $10^{-8}$ is recommended (doesn't affect much)

### Learning Rate Decay

One of the things that might help speed up your learning algorithm is to slowly reduce your learning rate over time. We call 
this **learning rate decay**. During the initial phases, you can afford to take much bigger steps, but as learning approaches 
converges, then having a slower learning rate allows you to take smaller steps.  

To implement this, you may use $\alpha = \frac{1}{1 + \text{decayRate} \times \text{epochNum} } \alpha_0$ and decay rate 
is a hyperparameter than you might need to tune. 

There are other ways to implement learning rate decay as well. 

$\alpha = 0.95^{epochNum} \alpha_0$ (exponential decay)

$\alpha = \frac{k}{\sqrt{epochNum}} \alpha_0$

$\alpha = \frac{k}{\sqrt{t}} \alpha_0$

Moreover, there is technique that decrease the learning rate in discrete steps, and you can also manually control the learning rate.

### Problem of Local Optima

In early days of deep learning, people used to worry a lot about the optimization algorithm getting stuck in bad **local optima**. 
But as this theory of deep learning has advanced, our understanding of local optima is also changing.

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl2203.png)

In the picture, it looks like there are a lot of local optimas in all those places, and it seems possible for algorithms to 
get stuck in a local optimum rather than find its way to a global optimum. In fact, this intuition isn't actually correct.

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl2204.png)

It turns out if you create a neural network, most points of zero gradient in a cost function are saddle points. Informally, 
a function of very high dimensional space, if the gradient is zero, a convex light function or a concave light function, and 
the chance of having local optima problem is extremely low.

![](https://raw.githubusercontent.com/evfox9/blog/master/deeplearning/dl2205.png)

It turns out that plateaus can really slow down learning, and a **plateau** is a region where the derivative is close to zero 
for a long time. This is where algorithms like momentum, RMSprop or Adam can really help your learning algorithm well.


## Programming Assignment

[Optimization](https://github.com/evfox9/Coursera/blob/master/Deep_Learning/Improving_Neural_Networks/Optimization.ipynb)

---
## References

[Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network)
