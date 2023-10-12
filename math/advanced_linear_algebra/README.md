## Linear Algebra
### Advanced Linear Algebra
#### 1. Matrix of Minors and Cofactors
##### 1.1. Matrix of Minors
###### 1.1.1. Definition
```text
The matrix of minors of a square matrix `A` is the matrix of determinants of the square submatrices of `A`.
``` 
```code
A = [a b c]
    [d e f]
    [g h i]
```
```text
The matrix of minors of `A` is
```
```code
M = [ei - fh  di - fg  dh - eg]
    [bi - ch  ai - cg  ah - bg]
    [bf - ce  af - cd  ae - bd]
```
```text
The matrix of minors of `A` is denoted by `M(A)`.
```
```text
The matrix of minors of `A` is the transpose of the matrix of cofactors of `A`.
```

##### 1.2. Matrix of Cofactors
###### 1.2.1. Definition
```text
The matrix of cofactors of a square matrix `A` is the matrix of signed minors of the square submatrices of `A`.
```
```code
A = [a b c]
    [d e f]
    [g h i]
```
```text
The matrix of cofactors of `A` is
```
```code
C = [+ei - fh  -di + fg  +dh - eg]
    [-bi + ch  +ai - cg  -ah + bg]
    [+bf - ce  -af + cd  +ae - bd]
```
```text
The matrix of cofactors of `A` is denoted by `C(A)`.
```
```text
The matrix of cofactors of `A` is the transpose of the matrix of minors of `A`.
```

#### 2. Adjugate Matrix
##### 2.1. Definition
```text
The adjugate matrix of a square matrix `A` is the transpose of the matrix of cofactors of `A`.
```
```code
A = [a b c]
    [d e f]
    [g h i]
```
```text
The adjugate matrix of `A` is
```
```code
adj(A) = [+ei - fh  -di + fg  +dh - eg]
         [-bi + ch  +ai - cg  -ah + bg]
         [+bf - ce  -af + cd  +ae - bd]
```
```text
The adjugate matrix of `A` is denoted by `adj(A)`.
```
```text
The adjugate matrix of `A` is the transpose of the matrix of cofactors of `A`.
```
##### 2.2. Properties
###### 2.2.1. Property
```text
Let `A` be a square matrix. Then
```
```code
A * adj(A) = adj(A) * A = det(A) * I
```
```text
where `I` is the identity matrix of the same size as `A`.
```

###### 2.2.3. Corollary
```text
Let `A` be a square matrix. Then
```
```code
A * adj(A) = adj(A) * A = det(A) * I
```
```text
where `I` is the identity matrix of the same size as `A`.
```
```text
If `A` is invertible, then
```
```code
A^-1 = (1 / det(A)) * adj(A)
```
```text
where `A^-1` is the inverse of `A`.
```

#### 3. Inverse Matrix
##### 3.1. Definition
```text
Let `A` be a square matrix. If there exists a square matrix `B` such that
```
```code
A * B = B * A = I
```
```text
where `I` is the identity matrix of the same size as `A`, then `B` is called the inverse of `A` and is denoted by `A^-1`.
```
```text
If `A` is invertible, then `A^-1` is unique.
```
##### 3.2. Properties
###### 3.2.1. Property
```text
Let `A` be a square matrix. Then
```
```code 
A * A^-1 = A^-1 * A = I
```
```text
where `I` is the identity matrix of the same size as `A`.
```

#### 4. Determinant of a Matrix
##### 4.1. Definition
```text
Let `A` be a square matrix. The determinant of `A` is a scalar denoted by `det(A)` or `|A|`.
```
```text
If `A` is a 2x2 matrix, then
```
```code
A = [a b]
    [c d]
```
```text
The determinant of `A` is
```
```code
det(A) = ad - bc
```
```text
If `A` is a 3x3 matrix, then
```
```code
A = [a b c]
    [d e f]
    [g h i]
```
```text
The determinant of `A` is
```
```code
det(A) = aei + bfg + cdh - ceg - bdi - afh
```
```text
If 'A' is a higher than 3x3 matrix, then
```
```text
The determinant of `A` is
```
```code
det(A) = a_11 * A_11 + a_12 * A_12 + ... + a_1n * A_1n
```
```text
where `a_11`, `a_12`, ..., `a_1n` are the elements of the first row of `A` and `A_11`, `A_12`, ..., `A_1n` are the cofactors of `a_11`, `a_12`, ..., `a_1n`.
```
```text
The determinant of `A` is the sum of the products of the elements of the first row of `A` and their cofactors.
```
##### 4.2. Properties
###### 4.2.1. Property
```text
Let `A` be a square matrix. Then
```
```code
det(A) = det(A^T)
```

###### 4.2.3. Property
```text
Let `A` and `B` be square matrices of the same size. Then
```
```code
det(A * B) = det(A) * det(B)
```

###### 4.2.5. Property
```text
Let `A` be a square matrix. Then
```
```code
det(A^-1) = 1 / det(A)
```

###### 4.2.7. Property
```text
Let `A` be a square matrix. Then
```
```code
det(k * A) = k^n * det(A)
```
```text
where `k` is a scalar and `n` is the size of `A`.
```

###### 4.2.9. Property
```text
Let `A` be a square matrix. Then
```
```code
det(A) = 0
```
```text
if and only if `A` is singular.
```

### Definites and Semidefinites
#### 1. Positive Definite
##### 1.1. Definition
```text
Let `A` be a square matrix. Then `A` is positive definite if and only if
```
```code
x^T * A * x > 0
```
```text
for all nonzero vectors `x`.
```

###### 1.2. Property
```text
Let `A` be a square matrix. Then `A` is positive definite if and only if
```
```code
A = A^T
```
```text
and
```
```code
det(A) > 0
```

##### 2. Negative Definite
###### 2.1. Definition
```text
Let `A` be a square matrix. Then `A` is negative definite if and only if
```
```code
x^T * A * x < 0
```
```text
for all nonzero vectors `x`.
```

###### 2.2. Property
```text
Let `A` be a square matrix. Then `A` is negative definite if and only if
```
```code
A = A^T
```
```text
and
```
```code
det(A) < 0
```

##### 3. Positive Semidefinite
###### 3.1. Definition
```text
Let `A` be a square matrix. Then `A` is positive semidefinite if and only if
```
```code
x^T * A * x >= 0
```
```text
for all nonzero vectors `x`.
```

###### 3.2. Property
```text
Let `A` be a square matrix. Then `A` is positive semidefinite if and only if
```
```code
A = A^T
```
```text
and
```
```code
det(A) >= 0
```

##### 4. Negative Semidefinite
###### 4.1. Definition
```text
Let `A` be a square matrix. Then `A` is negative semidefinite if and only if
```
```code
x^T * A * x <= 0
```
```text
for all nonzero vectors `x`.
```

###### 4.2. Property
```text
Let `A` be a square matrix. Then `A` is negative semidefinite if and only if
```
```code
A = A^T
```
```text
and
```
```code
det(A) <= 0
```

##### 5. Indefinite
###### 5.1. Definition
```text
Let `A` be a square matrix. Then `A` is indefinite if and only if
```
```code
x^T * A * x < 0
```
```text
for some nonzero vectors `x` and
```
```code
x^T * A * x > 0
```
```text
for some nonzero vectors `x`.
```

###### 5.2. Property
```text
Let `A` be a square matrix. Then `A` is indefinite if and only if
```
```code
A != A^T
```
```text
or
```
```code
det(A) = 0
```

### Eigenvalues and Eigenvectors
#### 1. Eigenvalues and Eigenvectors
##### 1.1. Definition
```text
Let `A` be a square matrix. A scalar `lambda` is called an eigenvalue of `A` if there exists a nonzero vector `x` such that
```
```code
A * x = lambda * x
```
```text
The vector `x` is called an eigenvector of `A` corresponding to `lambda`.
```
```text
If `A` is a 2x2 matrix, then
```
```code
A = [a b]
    [c d]
```
```text
The eigenvalues of `A` are the solutions of the equation
```
```code
det(A - lambda * I) = 0
```
```text
where `I` is the identity matrix of the same size as `A`.
```
```text
If `A` is a 3x3 matrix, then
```
```code
A = [a b c]
    [d e f]
    [g h i]
```
```text
The eigenvalues of `A` are the solutions of the equation
```
```code
det(A - lambda * I) = 0
```
```text
where `I` is the identity matrix of the same size as `A`.
```

##### 1.2. Properties
###### 1.2.1. Property
```text
Let `A` be a square matrix. Then
```
```code
det(A) = lambda_1 * lambda_2 * ... * lambda_n
```
```text
where `lambda_1`, `lambda_2`, ..., `lambda_n` are the eigenvalues of `A`.
```
