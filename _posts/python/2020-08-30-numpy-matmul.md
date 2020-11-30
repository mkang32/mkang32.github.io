---
layout: post
title:  "What Should I Use for Dot Product and Matrix Multiplication?: NumPy multiply vs. dot vs. matmul vs. @"
date:   2020-08-30 00:00:00 -0400
categories: python
---

When I first implemented gradient descent from scratch a few years ago, I was very confused which method to use for 
dot product and matrix multiplications - `np.multiply` or `np.dot` or `np.matmul`? And after a few years, it turns out that... I am 
still confused! So, I 
decided to investigate all the options in Python and NumPy 
(`*`, `np.multiply`, `np.dot`, `np.matmul`, and `@`), come up with the best approach to take, and document the findings here.  

TLDL; Use `np.dot` for dot product. For matrix multiplication, use `@` for Python 3.5 or above, and `np.matmul` for 
earlier versions.

# Table of contents
1. [What are dot product and matrix multiplications?](#dot_product)
2. [What is available for NumPy arrays?](#numpy_array)  
    (1) [element-wise multiplication: * and sum](#asterisk)  
    (2) [element-wise multiplication: np.multiply and sum](#np.multiply)  
    (3) [dot product: np.dot](#np.dot)  
    (4) [matrix multiplication: np.matmul](#np.matmul)  
    (5) [matrix multiplication: @](#@)  
3. [So.. what's with np.not vs. np.matmul (@)?](#dot_vs_matmul)  
4. [Summary](#summary)  
5. [Reference](#reference)

<a id="dot_product"></a>
# 1. What are dot product and matrix multiplication?

If you are not familiar with dot product or matrix multiplication yet or if you need a quick recap, check out the 
previous blog post: <a href="{{site.baseurl}}/python/2020/08/23/dot-product.html">What are dot product and 
matrix multiplication?</a> 

In short, the dot product is the sum of products of values in two same-sized vectors and the matrix multiplication 
is a 
matrix 
version 
of the dot product with two matrices. The output of the dot product is a scalar whereas that of the matrix 
multiplication 
is a matrix whose
 elements are the dot products of pairs of vectors in each matrix. 

Dot product: 

$$
[a_1 \ a_2]
\begin{bmatrix}
b_1 \\
b_2 
\end{bmatrix}
=a_1b_1 + a_2b_2
$$


Matrix multiplication: 

$$
\begin{bmatrix}
a_{11} \ \ a_{12} \\
a_{21} \ \ a_{22} \\
\end{bmatrix}

\begin{bmatrix}
b_{11} \ \ b_{12} \\
b_{21} \ \ b_{22} \\
\end{bmatrix}

=
\begin{bmatrix}
a_{11}b_{11} + a_{12}b_{21} \ \ \ a_{11}b_{12} + a_{12}b_{22}\\
a_{21}b_{11} + a_{22}b_{21} \ \ \ a_{21}b_{12} + a_{22}b_{22}\\
\end{bmatrix}
$$

<br>

<a id="numpy_array"></a>
# 2. What's available for NumPy arrays? 

So, there are multiple options you can use to perform dot product or matrix multiplication:

1. basic element-wise multiplication: `*` or `np.multiply` along with `np.sum`
2. dot product: `np.dot` 
3. matrix multiplication: `np.matmul`, `@`  

We will go through different scenarios depending on the dimensions of vectors/matrices and understand the pros and cons
 of each method. To run the code in the following sections, We first need to import numpy.  


```python
import numpy as np
```

<a id="asterisk"></a>
## (1) element-wise multiplication: * and sum

First, we can try the fundamental approach using element-wise multiplication based on the definition of dot product: 
multiply corresponding
elements in two vectors and then sum all the output values. The downside of this approach is that you need 
**separate operations
 for product and sum** and it is **slower** than other methods we will discuss later. 

Here is an example of dot product with two 1D arrays.
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

>>> a*b
array([ 4, 10, 18])

>>> sum(a*b)
32
```


Can we use the same `*` and `sum` operation for matrix multiplication? Let's check out.


```python
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([1, 1, 1])

>>> c*d
array([[1, 2, 3],
       [4, 5, 6]])
       
>>> sum(c*d)
array([5, 7, 9])
```

Wait, it looks different from what we would get from our own calculation below!

$$
\begin{bmatrix}
1 & 2 & 3  \\
4 & 5 & 6
\end{bmatrix}
\begin{bmatrix}
1 \\
1 \\
1 \\
\end{bmatrix} = 
\begin{bmatrix}
1 \times 1 + 2 \times 1 + 3 \times 1 \\
4 \times 1 + 5 \times 1 + 6 \times 1
\end{bmatrix}
=\begin{bmatrix}
6  \\
15
\end{bmatrix}
$$

So, it turns out that we need to be careful when we apply `sum` after `*` operation.
 
 Let's look at it step by step. Here is what happened at 
`c*d`. Each 
row of 2D array $c$ is 
considered as
 an element of the matrix and it is 
paired with the second array $d$ 
for element-wise multiplication.  

$$
\begin{bmatrix}
[1 & 2 & 3] * [1 & 1 & 1]  \\
[4 & 5 & 6] * [1 & 1 & 1] 
\end{bmatrix} = 
\begin{bmatrix}
[1 \times 1 & 2 \times 1 & 3 \times 1] \\
[4 \times 1 & 5 \times 1 & 6 \times 1]
\end{bmatrix}
=\begin{bmatrix}
1 \ 2 \ 3  \\
4 \ 5 \ 6 
\end{bmatrix}
$$

And then, when we apply `sum`, the Python's default `sum` function takes all the element in a NumPy array at once, 
which 
became 
$1+2+ ..
.+ 5+6 = 21$. But what we want is to sum only elements in each row. So we need to find an alternative to `sum`. 

Here comes `np
.sum` to rescue. When we pass the parameter `axis=1`, it sums elements across columns in the same row. The default
 is 
`axis=0` 
which 
sums elements across rows within the same column, so we need to make sure we pass `axis=1` parameter.


```python
>>> np.sum(c*d, axis=1)
array([ 6, 15])
```

Yes! This is what we expected.


<a id="np.multiply"></a>
## (2) element-wise multiplication: np.multiply and sum
Okay, then what about `np.multiply`? What does it do and is it different from `*`? 
 
`np.multiply` is basically the same as `*`. It is a `NumPy`'s version of element-wise 
multiplication instead of Python's native operator. 


```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

>>> c = np.multiply(a, b)
>>> c 
array([ 4, 10, 18])

>> np.sum(c, axis=1)
array([ 6, 15])

```



<a id="np.dot"></a>
## (3) dot product: np.dot
Is there any option that we can avoid the additional line of `np.sum`? Yes, `np.dot` in NumPy! You 
can use either `np.dot(a, b)` or `a.dot(b)` and 
it **takes care of both element multiplication and sum**. Simple and easy.


```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

>>> np.dot(a, b)
32
```




Great! Dot product in just one line of code. If 
the 
dimension
 of the array is 2D
 or higher, make sure the number of columns of the first array matches up with the number of rows in the second array. 


```python
a = np.array([[1, 2, 3]])  # shape (1, 3)
b = np.array([[4, 5, 6]])  # shape (1, 3)

>>> np.dot(a, b)  
# ValueError: shapes (1,3) and (1,3) not aligned: 3 (dim 1) != 1 (dim 0)
```


To make the above example work, you need to transpose the second array so that the shapes are aligned: (1, 3) x (3, 
1). Note that this will return (1, 1), which is a 2D array.


```python
a = np.array([[1, 2, 3]])  # shape (1, 3)
b = np.array([[4, 5, 6]])  # shape (1, 3)

>>> np.dot(a, b.T)  
array([[32]])
```

As a side note, if you transpose the second array, you will get a (3 x 3) array, which is the outer product instead of 
inner product (dot product). So, be make sure you transpose the right one.


Now let's try a 2D x 2D example as well with the following example. Will it work even if it's called `dot` product? 

$$
\begin{bmatrix}
1, 2, 3 \\
4, 5, 6 
\end{bmatrix}
\begin{bmatrix}
1 \\
1 \\
1 \\
\end{bmatrix} =
\begin{bmatrix}
1 \times 1 + 2 \times 1 + 3 \times 1 \\
4 \times 1 + 5 \times 1 + 6 \times 1 \\
\end{bmatrix} =
\begin{bmatrix}
6 \\
15 \\
\end{bmatrix}
$$


```python
c = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
d = np.array([[1], [1], [1]])  # shape (3, 1)

>>> np.dot(c, d)
array([[ 6],
       [15]])
```



It works! Even if it is called `dot`, which indicates that the inputs are 1D vectors and the output is a scalar by 
its definition, it works for 2D or higher dimensional matrices as if it was a matrix multiplication. 

So, should we use `np.dot` for both dot product and matrix multiplication?

Technically yes but it is not recommended to use `np.dot` for matrix multiplication because the name dot product 
has a specific meaning and it can be confusing to readers, especially mathematicians! [(Reference)](https://blog
.finxter.com/numpy-matmul-operator/#Python_@_Operator) Also, it is not recommended for high dimensional matrices (3D 
or above) because `np.dot` behaves different from 
normal matrix multiplication. We will discuss this in the later of this post. 

So, **`np.dot` works for both dot product and matrix multiplication but is recommended for dot product only.** 

<a id="np.matmul"></a>
## (4) matrix multiplication: np.matmul
The next option is `np.matmul`. **It is designed for matrix multiplication** and even the name comes from it 
(**MAT**rix **MUL**tiplication). Although the name says matrix multiplication, it also works in 1D array and can do 
dot product 
just like 
`np.dot`. 


```python
# 1D array
a = np.array([1, 2, 3])  # shape (1, 3)
b = np.array([4, 5, 6])  # shape (1, 3)

>>> np.matmul(a, b)
32
```


```python
# 2D array with values in 1 axis
a = np.array([[1, 2, 3]])  # shape (1, 3)
b = np.array([[4, 5, 6]])  # shape (1, 3)

>>> np.dot(a, b.T) 
array([[32]])
```


```python
# two 2D arrays
c = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
d = np.array([[1], [1], [1]])  # shape (3, 1)

>>> np.dot(c, d)
array([[ 6],
       [15]])
```


Nice! So, this means both `np.dot` and `np.matmul` work perfectly for dot product and matrix multiplication. However,
 as we said before, **it is recommended to use `np.dot` for dot product and `np.matmul` for 2D or higher matrix 
 multiplication**.

<a id="@"></a>
## (5) matrix multiplication: @

Here comes our last but not least option, `@`! `@`, pronounced as [at], is a new Python operator that was introduced 
since 
Python 3.5, 
whose name comes 
from m**AT**rices. **It is basically the same as `np.matmul` and designed to perform matrix multiplication**. But why do
 we need a new infix if we already have `np.matmul` that works perfectly fine? 

The major motivation for adding a new operator to stdlib was that the matrix multiplication is a so common operator that it deserves its own infix. For example, the operator `//` is much more uncommon than matrix multiplication but still has its own infix. To learn more about the background of this addition, check out this [PEP 465](https://www.python.org/dev/peps/pep-0465/).


```python
# 1D array
a = np.array([1, 2, 3])  # shape (1, 3)
b = np.array([4, 5, 6])  # shape (1, 3)

>>> a @ b  
32
```




```python
# 2D array with values in 1 axis
a = np.array([[1, 2, 3]])  # shape (1, 3)
b = np.array([[4, 5, 6]])  # shape (1, 3)

>>> a @ b.T
array([[32]])
```



```python
# 2D arrays
c = np.array([[1, 2, 3], [4, 5, 6]])  # shape: (2, 3)
d = np.array([[1], [1], [1]])  # shape: (3, 1)

>>> c @ d
array([[ 6],
       [15]])
```



So, **`@` works exactly same as `np.matmul`**. But which one should you use between `np.matmul` and `@` then? Although
 it is your preference, `@` looks cleaner than `np.matmul` in code. Let us see a case where have three matrices $x, 
 y, z$ to perform a matrix 
 multiplication.

```python
# `np.matmul` version
np.matmul(np.matmul(x, y), z)

# `@` version
x @ y @ z
```

As you can see, **`@` is much cleaner and more readable. However, as it is available only Python 3.5+, you have to use
 `np
.matmul` if you use an earlier Python version**.

<a id="dot_vs_matmul"></a>
## 3. So.. what's with np.dot vs. np.matmul (@)?

In the above section, I mentioned that `np.dot` is not recommended for high dimensional arrays. What do I mean by 
that?  

There was an interesting [question](https://stackoverflow.com/questions/34142485/difference-between-numpy-dot-and-python-3-5-matrix-multiplication) in stackoverflow about different behaviors between `np.dot` and `@`. Let's looks at this.


```python
# define input arrays
a = np.random.rand(3,2,2)  # 2 rows, 2 columns, in 3 layers 
b = np.random.rand(3,2,2)  # 2 rows, 2 columns, in 3 layers 

# perform matrix multiplication
c = np.dot(a, b)
d = a @ b  # Python 3.5+

>>> c.shape  # np.dot
(3, 2, 3, 2)

>>> d.shape  # @
(3, 2, 2)
```
 
With the same inputs, we have completely different outputs - 4D array for `np.dot` and
 3D 
array for `@`.
 What happened? This is 
because of 
the way `np.dot` and `@` are designed. Based on the their definition: 

=======================  
For `matmul`:
> If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.

For `np.dot`:
> For 2-D arrays it is equivalent to matrix multiplication, and for 1-D arrays to inner product of vectors (without complex conjugation). For N dimensions it is a sum product over the last axis of a and the second-to-last of b

> If a is an N-D array and b is an M-D array (where M>=2), it is a sum product over the last axis of a and the second-to-last axis of b:  

> $ dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])$

=======================  

**Long story short, in the normal matrix multiplication situation where we want to treat each stack of matrices in the 
last two indexes, we should use `matmul`**. 


<a id="summary"></a>
# 4. Summary

- `*` == `np.multiply` != `np.dot` != `np.matmul` == `@`
- `*` and `np.multiply` need `np.sum` to perform dot product. Not recommended for dot product or matrix multiplication.
- `np.dot` works for dot product and matrix multiplication. However, recommended to avoid using it for matrix multiplication due to the name. 
- `np.matmul` and `@` are the same thing, designed to perform matrix multiplication. `@` is added to Python 3.5+ to give matrix multiplication its own infix. 
- `np.dot` and `np.matmul` generally behave similarly other than 2 exceptions: 1) `matmul` doesn't allow multiplication by scalar, 2) the calculation is done differently for N>2 dimesion. Check the documentation which one you intend to use. 

One line summary: 

- **For dot product, use `np.dot`. For matrix multiplication, use `@` for Python 3.5 or above, and `np.matmul` for earlier Python versions.**   

<a id="reference"></a>
# 5. Reference
- [NumPy Matrix Multiplication — np.matmul() and @](https://blog.finxter.com/numpy-matmul-operator/)
- [numpy.dot official document](https://numpy.org/doc/stable/reference/generated/numpy.dot.html)
- [PEP 465 -- A dedicated infix operator for matrix multiplication](https://www.python.org/dev/peps/pep-0465/)
- [Difference between numpy dot() and Python 3.5+ matrix multiplication @](https://stackoverflow.com/questions/34142485/difference-between-numpy-dot-and-python-3-5-matrix-multiplication)
- [Wikipedia](https://en.wikipedia.org/wiki/Dot_product)

