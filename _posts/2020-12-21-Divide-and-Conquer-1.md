---
title: 1-1. Divide and Conquer, Sorting and Searching, and Randomized Algorithms (1)
tags: Computer_Science Algorithm Coursera Algorithm_Specialization
---

## Introduction

### Why Study Algorithms?

**Algorithm** is a set of well-defined rules, a recipe in effect for solving some computational problem. So why study algorithms?

- important for all other branches of computer science

- plays a key role in modern technological innovation

- provides novel "lens" on processes outside of computer science and technology

- challenging

- fun

### Integer Multiplication

Input: two $n$-digit numbers $x$ and $y$

Output: the product $xy$

![](https://raw.githubusercontent.com/evfox9/blog/master/algorithms/al1101.png)

This is the integer multiplication algorithm that we learned in school. Number of operations overall grows like $c n^2$ for some constant $c$ and input length $n$.

Can we do better?

### Karatsuba's Multiplication

![](https://raw.githubusercontent.com/evfox9/blog/master/algorithms/al1102.png)

This method is called **Karatsuba's multiplication**.

### Merge Sort

Input: array of $n$ numbers, unsorted

Output: same numbers, sorted in increasing order

![](https://raw.githubusercontent.com/evfox9/blog/master/algorithms/al1103.png)

#### Pseudocode for Merge

C = output array [length = $n$]
A = 1st sorted array [$\frac{n}{2}$]
B = 2nd sorted array [$\frac{n}{2}$]



``` cpp
i = 1, j = 1

for k = 1 to n
  if A(i) < B(j)
    C(k) = A(i)
    i++
  else [B(j) < A(i)]
    C(k) = B(j)
    j++
end
```


#### Running time of Merge Sort

Claim: Merge Sort requires $\leq 6n \log_{2}{n} + 6n$ operations to sort $n$ numbers.

Proof: Assuming $n =$ power of $2$, we will use recursion tree. Then, this recursion tree will have roughly $\log_{2}{n}$ number of levels and
at each level $j=0,1,2,\cdots,\log_{2}{n}$, there are $2^j$ subproblems, each of size $\frac{n}{2^j}$.
Total number of operations at level $j$ will be at most $2^j \times 6 (\frac{n}{2^j}) = 6n$.
Therefore, the overall upper bound for the merge sort will be $6n (\log_{2}{n}+1)$.

### Guiding Principles for Analysis of Algorithms

#### Guiding Principle #1

Worst-case analysis: our running time bound holds for every input of length n, particularly appropriate for general purpose routines.
As opposed to average-case analysis and benchmarks, which requires domain knowledge.

#### Guiding Principle #2

Won't pay much attention to constant factors, lower-order terms, because

- way easier

- constants depend on architecture/compiler/programmer anyway

- lose very little predictive power

### Guiding Principle #3

Asymptotic analysis: focus on running time for large input sizes $n$.

E.g.: $6n \log_{2}{n} + 6n$ is better than $\frac{1}{2}n^2$

![](https://raw.githubusercontent.com/evfox9/blog/master/algorithms/al1104.png)

Definition of a "fast algorithm" in this course is adopt these three biases as guiding principles.

## Asymptotic Analysis

### The Gist

Importance: vocabulary for the design and analysis of algorithms (e.g. "big-oh" notation)

- "sweet spot" for high-level reasoning about algorithms

- coarse enough to suppress architecture/language/compiler-dependent details

- sharp enough to make useful comparisons between different algorithms, especially on large inputs

High-level idea: suppress constant factors and lower-order terms.

Example: equate $6n \log_{2}{n} + 6n$ with just $n \log{n}$.

Terminology: running time is $O(n \log{n})$.

---

Problem: does array $A$ contain the integer $t$?

given $A$ (array of length $n$) and $t$ (an integer)

``` cpp
for i = 1 to n
  if A[i] == t return TRUE
return FALSE
```

Question: What is the running time?

The answer is $O(n)$.

---

Problem: do arrays $A, B$ contain the integer $t$?

given $A,B$ (array of length $n$) and $t$ (an integer)

``` cpp
for i = 1 to n
  if A[i] == t return TRUE
for i = 1 to n
  if B[i] == t return TRUE
return FALSE
```

Question: What is the running time?

The answer is $O(n)$.

---

Problem: do arrays $A, B$ have a number in common?

given arrays $A,B$ of length $n$

``` cpp
for i = 1 to n
  for j = 1 to n
    if A[i] == B[j] return TRUE
return FALSE
```

Question: What is the running time?

The answer is $O(n^2)$.

---

Problem: does array A have duplicate entries?

given array $A$ of length $n$

``` cpp
for i = 1 to n
  for j = i+1 to n
    if A[i] == A[j] return TRUE
return FALSE
```

Question: What is the running time?

The answer is $O(n^2)$.


### Big-Oh Notation

>Definition: $T(n) = O(f(n))$ iff there exist constants
$c, n_0 > 0 \ \text{s.t} \ T(n) \leq c \cdot f(n)$ for $\forall n \geq n_0$.

---

Claim: if $T(n) = a_k n^k + \cdots + a_1 n + a_0$, then $T(n) = O(n^k)$.

Proof: Choose $n_0 = 1$ and $c = | a_k | + | a_{k-1} | + \cdots + | a_1 | + | a_0 |$.

$\forall n \geq 1$,

$$T(n) \leq | a_k | n^k + \cdots + | a_1 | n + | a_0 | $$

$$ \leq | a_k | n^k + \cdots + | a_1 | n^k + | a_0 | n^k $$

$$ = c \cdot n^k$$

---

Claim: $\forall k \geq 1,\ n^k \neq O(n^{k-1})$.

Proof: (by contradiction) Suppose $n^k = O(n^{k-1})$. Then, $\exists$ constant $c, n_0 \ \text{s.t} \ n^k \leq c \cdot n^{k-1}$ for $\forall n \geq n_0$.

But canceling $n^{k-1}$ from both sides leaves $n \leq c$ for $\forall n \geq n_0$, which is clearly false.

Therefore, $n^k \neq O(n^{k-1})$.

### Big Omega and Theta

> Definition: $T(n) = \Omega(f(n))$ iff $\exists$ constants $c, n_0 \ \text{s.t} \ T(n) \geq c \cdot f(n) \ \forall n \geq n_0$

> Definition: $T(n) = \Theta(f(n))$ iff $T(n) = O(f(n))$ and $T(n) = \Omega(f(n))$.
>
> Equivalent: $\exists$ constants $c_1, c_2, n_0 \ \text{s.t} \ c_1 f(n) \leq T(n) \leq c_2 f(n) \ \forall n \geq n_0$.

Question: Let $T(n) = \frac{1}{2}n^2 + 3n$. Which of the following statements are true? (check all that apply)

- $T(n) = O(n)$

- $T(n) = \Omega(n)$

- $T(n) = \theta(n^2)$

- $T(n) = O(n^3)$

The answer is $T(n) = \Omega(n)$, $T(n) = \theta(n^2)$ and $T(n) = O(n^3)$.

> Definition: $T(n) = o(f(n))$ iff $\forall$ constants $c > 0,\ \exists n_0 \ \text{s.t} \ T(n) \leq c \ \forall n \geq n_0$.

$\forall k \geq 1,\ n^{k-1} = o(n^k)$.

---

Claim: $2^{n+10} = O(2^n)$

Proof: Since $2^{n+10} = 1024 \cdot 2^n$, if we choose $c=1024$ and $n_0=1$, then $2^{n+10} \leq c \cdot 2^n \ \forall n \geq n_0$.

---

Claim: $2^{10n} \neq O(2^n)$

Proof: (by contradiction) If $2^{10n} = O(2^n)$, then $\exists$ constants $c, n_0 > 0 \ \text{s.t} \
2^{10n} \leq c \cdot 2^n \ \forall n \geq n_0$ .

But canceling $2^n$ from both sides leaves $2^{9n} \leq c \ \forall n \geq n_0$, which is certainly false.

---

Claim: For every pair of positive function of $f(n), g(n)$, $\max\{ f, g \} = O(f(n)+g(n))$.

Proof: $\forall n$, $\max \{ f(n), g(n) \} \leq f(n) + g(n)$ and $2\max \{ f(n), g(n) \} \geq f(n) + g(n)$.

$\therefore \frac{1}{2}(f(n)+g(n)) \leq \max \{ f(n), g(n) \} \leq f(n) + g(n) \ \forall n \geq 1$.

$\max \{ f(n), g(n) \} = O(f(n)+g(n))$ where $n_0 = 1, c_1 = \frac{1}{2}, c_2 = 1$.

---

### References

[Divide and Conquer, Sorting and Searching, and Randomized Algorithms](https://www.coursera.org/learn/algorithms-divide-conquer)
