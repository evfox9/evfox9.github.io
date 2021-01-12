---
title: 1-2. Divide and Conquer, Sorting and Searching, and Randomized Algorithms (2)
tags: Computer_Science Algorithm Coursera Algorithm_Specialization
---

## Divide & Conquer Algorithms

### The Divide and Conquer Paradigm

1. **Divide** into smaller subproblems.

2. **Conquer** via recursive calls.

3. **Combine** solutions of subproblems into one for the original problem.

### O(n log n) Algorithm for Counting Inversions

Input: array $A$ containing the numbers $1, 2, 3, \cdots, n$ in some arbitrary order.

Output: number of inversions = number of pairs $(i, j)$ of array indices with $i < j$ and $A[i] > A[j]$.

---

Question: What is the largest possible number of inversions that a 6-element array can have?

The answer is $15$. In general, $\left(\begin{array}{cc} n \\ 2 \end{array} \right) = \frac{n(n-1)}{2}$.

---

By brute-force algorithm, it can be done in $O(n^2)$ time. Can we do better? Yes, by using divide and conquer.

Call an inversion $(i,j)$ with $[i < j]$:

left if $i,j \leq \frac{n}{2}$

right if $i,j > \frac{n}{2}$

split if $i \leq \frac {n}{2} < j$


``` cpp
Count(array A, length n)
  if n==1 return 0
  else
    x = Count(1st half of A, n/2)
    y = Count(2nd half of A, n/2)
    z = CountSplitInv(A, n) # currently unimplemented
  return x+y+z
```

Our goal is to implement `CountSplitInv` in linear ($O(n)$) time. Then, `Count` will run in $O(n \log{n})$ time, just like merge sort.



### Strassen's Subcubic Matrix Multiplication Algorithm

Matrix multiplication that we know: $XY = Z$ where $z_{ij} =$ ($i$th row of $X$) $\cdot$ ($j$th row of $Y$).

Question: What is the asymptotic running time of the straightforward iterative algorithm for matrix multiplication?

The answer is $\theta(n^3)$.

---

Can we improve it? We can think of applying divide and conquer.

Write $X = \begin{pmatrix} A & B \\ C & D \end{pmatrix}$ and $Y = \begin{pmatrix} E & F \\ G & H \end{pmatrix}$, where $A, \cdots, H$ are all $\frac{n}{2} \times \frac{n}{2}$ sized matrices.

Then, $X \cdot Y = \begin{pmatrix} AE+BG & AF+BH \\ CE+DG & CF+DH \end{pmatrix}$.

Step 1: recursively compute the 8 necessary products.

Step 2: do the necessary additions.

Run time is $O(n^2)$.

---

Strassen's Subcubic Matrix Multiplication Algorithm

Step 1: recursively compute only 7 (cleverly chosen) products

Step 2: do the necessary (clever) additions + subtractions (still $O(n^2)$ time)

The seven products: $P_1 = A(F-H),\ P_2 = (A+B)H,\ P_3 = (C+D)E,\ P_4 = D(G-E),\ P_5 = (A+D)(E+H), P_6 = (B-D)(G+H), P_7 = (A-C)(E+F)$



## The Master Method

### Formal Statement

The **Master Method** is a "black box" for solving recurrences. This method is based on the assumption that all subproblems have equal size.

> Base case: $T(n) \leq a$ for all sufficiently small $n$
>
> For all larger $n$: $T(n) \leq aT(\frac{n}{b}) + O(n^d)$ where
>
>$a$ = number of recursive calls ($\geq 1$)
>
>$b$ = input size shrinkage factor ($> 1$)
>
>$c$ = exponent in running time of "combine step" ($\geq 0$)
>
> $T(n) = \begin{cases} O(n^d \log{n}) && \text{if} \ a = b^d \ \text{(case 1)}
\\ O(n^d) && \text{if} \ a < b^d \ \text{(case 2)} \\ O(n^{\log_{b}{a}}) && \text{if} \ a > b^d \ \text{(case 3)} \end{cases}$
>

### Example

Merge sort: $a = 2,\ b = 2,\ d = 1$

Since $a = b^d$, $T(n) \leq O(n^d \log{n}) = O(n\log{n})$.

---

Question: Where are the respective values of $a,\ b,\ d$ for a binary search of a sorted array, and which case of the Master Method does this correspond to?

The answer is $1,\ 2,\ 0$.

---

Question: Where are the respective values of $a,\ b,\ d$ for Gauss's recursive integer multiplication algorithm, and which case of the Master Method does this correspond to?

The answer is $3,\ 2,\ 1$.

---

Strassen's Matrix Multiplication algorithm: $a = 7,\ b = 2,\ d = 2$.

$T(n) = O(n^{\log_{2}{7}}) = O(n^{2.81})$

---

### References

[Divide and Conquer, Sorting and Searching, and Randomized Algorithms](https://www.coursera.org/learn/algorithms-divide-conquer)
