# Linea.rs
A basic rust linear algebra crate for personal use
## Project Structure

### Matrix
The `Matrix` object provides the following methods:
- `new`
- `from_vec`
- `fill`
- `get`
- `set`
- `row`
- `col`
- `is_square`
- `add`
- `add_scalar`
- `multiply`
- `multiply_scalar`
- `multiply_vector`
- `transpose`
- `determinant`
- `inverse`
- `lu_decompose`
- `qr_decompose`

#### Arithmetic Operators
To help make arithmetic easier, the following operators are implemented via `std::ops` traits:
| Expression          | Meaning                                         |
| ------------------- | ----------------------------------------------- |
| `&Matrix + &Matrix` | Matrix addition                                 |
| `&Matrix + f64`     | Add a scalar to each element of the matrix      |
| `&Matrix * &Matrix` | Matrix multiplication                           |
| `&Matrix * f64`     | Multiply each element of the matrix by a scalar |
| `&Matrix * &Vector` | Multiply a matrix by a vector                   |

**NOTE:** All operators panic if dimensions are incompatible. For safe, fallible operations, use the corresponding methods (e.g. `Matrix::add(&other) -> Result<Matrix, MatrixOperationError>`).

## Vector
The `Vector` object provides the following methods:
- `from_vec`
- `get`
- `set`
- `add`
- `sub`
- `multiply_scalar`
- `dot`

#### Arithmetic Operators
To help make arithmetic easier, the following operators are implemented via `std::ops` traits:
| Expression          | Meaning                                         |
| ------------------- | ----------------------------------------------- |
| `&Vector + &Vector` | Vector addition                                 |
| `&Vector - &Vector` | Vector subtraction                              |
| `&Vector * &Vector` | Vector dot product                              |
| `&Vector * f64`     | Multiply each element of the vector by a scalar |

**NOTE:** All operators panic if dimensions are incompatible. For safe, fallible operations, use the corresponding methods (e.g. `Vector::add(&other) -> Result<Vector, VectorOperationError>`).
