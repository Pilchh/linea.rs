use crate::{
    matrix::{Matrix, errors::MatrixOperationError},
    vector::Vector,
};

impl Matrix {
    /// Adds two matrices together
    ///
    /// # Arguments
    /// * `other` - the the matrix to add
    ///
    /// # Returns
    /// A new `Matrix` with the computed value.
    ///
    /// # Example
    /// ```
    /// use linears::matrix::Matrix;
    /// let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    ///
    /// let result = a.add(&b).unwrap();
    ///  
    /// assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
    /// ```
    pub fn add(&self, other: &Matrix) -> Result<Matrix, MatrixOperationError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixOperationError::InvalidDimensions {
                a_rows: self.rows,
                a_cols: self.cols,
                b_rows: other.rows,
                b_cols: other.cols,
            });
        }

        let size = self.rows * self.cols;

        let mut result_matrix = Matrix {
            rows: self.rows,
            cols: self.cols,
            data: Vec::with_capacity(size),
        };

        for index in 0..size {
            result_matrix
                .data
                .push(self.data[index] + other.data[index]);
        }

        Ok(result_matrix)
    }

    /// Multiplies two matrices together
    ///
    /// # Arguments
    /// * `other` - the the matrix to multiply with
    ///
    /// # Returns
    /// A new `Matrix` with the computed value.
    ///
    /// # Example
    /// ```
    /// use linears::matrix::Matrix;
    /// let a = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// let b = Matrix::from_vec(2, 4, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]).unwrap();
    ///
    /// let result = a.multiply(&b).unwrap();
    ///     
    /// assert_eq!(
    ///     result.data,
    ///     vec![
    ///         29.0, 32.0, 35.0, 38.0, 65.0, 72.0, 79.0, 86.0, 101.0, 112.0, 123.0, 134.0
    ///     ]
    /// );
    /// ```
    pub fn multiply(&self, other: &Matrix) -> Result<Matrix, MatrixOperationError> {
        if self.cols != other.rows {
            return Err(MatrixOperationError::InvalidDimensions {
                a_rows: self.rows,
                a_cols: self.cols,
                b_rows: other.rows,
                b_cols: other.cols,
            });
        }

        let mut result_matrix = Matrix::new(self.rows, other.cols);
        result_matrix.fill(0.0);

        // matrix[i][j] = matrix[i * cols + j]
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;

                for k in 0..self.cols {
                    sum += self[(i, k)] * other[(k, j)];
                }

                result_matrix[(i, j)] = sum;
            }
        }

        Ok(result_matrix)
    }

    /// Multiplies a matrix and a vector
    ///
    /// # Arguments
    /// * `vector` - the vector to multiply by
    ///
    /// # Returns
    /// A new `Vector` with the computed value.
    ///
    /// # Example
    /// ```
    /// use linears::matrix::Matrix;
    /// use linears::vector::Vector;
    /// let mut matrix = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let vector = Vector::from_vec(vec![2.0, 4.0]);
    ///
    /// let column = matrix.mul_vector(&vector);
    /// ```
    pub fn mul_vector(&self, vector: &Vector) -> Result<Vector, MatrixOperationError> {
        if self.cols != vector.size {
            return Err(MatrixOperationError::MatrixVectorInvalidDimensions {
                matrix_rows: self.rows,
                matrix_cols: self.cols,
                vector_size: vector.size,
            });
        }

        let mut result_vector = Vector {
            size: vector.size,
            data: vec![0.0; self.rows],
        };

        for i in 0..self.rows {
            let mut sum = 0.0;

            for j in 0..self.cols {
                sum += self[(i, j)] * vector[j];
            }

            result_vector[i] = sum;
        }

        Ok(result_vector)
    }

    /// Adds a scalar value to all values in the matrix
    ///
    /// # Arguments
    /// * `scalar` - the value to add
    ///
    /// # Returns
    /// A new `Matrix` with the computed value.
    ///
    /// # Example.collect();
    /// ```
    /// use linears::matrix::Matrix;
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let mut matrix = Matrix::from_vec(2, 2, data).unwrap();
    ///
    /// let column = matrix.scalar_add(10.0);
    /// ```
    pub fn scalar_add(&self, scalar: f64) -> Matrix {
        self.map(|value| value + scalar)
    }

    /// Multiplies all values in a matrix by a scalar value
    ///
    /// # Arguments
    /// * `scalar` - the value to multiply by
    ///
    /// # Returns
    /// A new `Matrix` with the computed value.
    ///
    /// # Example
    /// ```
    /// use linears::matrix::Matrix;
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let mut matrix = Matrix::from_vec(2, 2, data).unwrap();
    ///
    /// let column = matrix.scalar_multiply(10.0);
    /// ```
    pub fn scalar_multiply(&self, scalar: f64) -> Matrix {
        self.map(|value| value * scalar)
    }

    /// Transpose the matrix
    ///
    /// # Returns
    /// A new `Matrix` with transposed value.
    ///
    /// # Example
    /// ```
    /// use linears::matrix::Matrix;
    /// let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    ///
    /// let result = a.transpose();
    ///
    /// assert_eq!(result.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    /// ```
    pub fn transpose(&self) -> Matrix {
        let mut result_matrix = Matrix::new(self.cols, self.rows);
        result_matrix.fill(0.0);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result_matrix[(j, i)] = self[(i, j)];
            }
        }

        result_matrix
    }
}

impl std::ops::Add for &Matrix {
    type Output = Matrix;

    fn add(self, other: Self) -> Matrix {
        self.add(other).unwrap()
    }
}

impl std::ops::Mul for &Matrix {
    type Output = Matrix;

    fn mul(self, other: Self) -> Matrix {
        self.multiply(other).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_addition() {
        // Arrange
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        // Act
        let result = &a + &b;

        assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_matrix_multiplication() {
        // Arrange
        let a = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Matrix::from_vec(2, 4, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]).unwrap();

        // Act
        let result = &a * &b;

        // Assert
        assert_eq!(result.rows, 3);
        assert_eq!(result.cols, 4);
        assert_eq!(
            result.data,
            vec![
                29.0, 32.0, 35.0, 38.0, 65.0, 72.0, 79.0, 86.0, 101.0, 112.0, 123.0, 134.0
            ]
        );
    }

    #[test]
    fn test_matrix_scalar_add() {
        // Arrange
        let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Act
        let scalar_matrix = m.scalar_add(10.0);

        // Assert
        assert_eq!(scalar_matrix.data, vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_matrix_scalar_multiply() {
        // Arrange
        let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Act
        let scalar_matrix = m.scalar_multiply(10.0);

        // Assert
        assert_eq!(scalar_matrix.data, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_matrix_vec_multiplication() {
        // Arrange
        let m = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let v = Vector::from_vec(vec![10.0, 20.0]);

        // Act
        let result = m.mul_vector(&v).unwrap();

        // Assert
        assert_eq!(result.data, vec![50.0, 110.0, 170.0]);
    }

    #[test]
    fn test_matrix_transposition() {
        // Arrange
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Act
        let result = m.transpose();

        // Assert
        assert_eq!(result.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
