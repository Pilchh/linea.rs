use crate::math::errors::{MatrixCreationError, MatrixOperationError};

// Row-major implementation of a matrix
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    /// Creates a new Matrix filled with provided value
    ///
    /// # Arguments
    /// * `rows` - the number of rows
    /// * `cols` - the number of columns
    /// * `n` - the value to fill the matrix with
    ///
    /// # Returns
    /// A new `Matrix` with defined properties filled with the value n.
    ///
    /// # Example
    /// ```
    /// use linears::math::matrix::Matrix;
    /// let matrix = Matrix::new_filled(2, 2, 0.0);
    /// ```
    pub fn new_filled(rows: usize, cols: usize, n: f64) -> Result<Matrix, MatrixCreationError> {
        if rows == 0 || cols == 0 {
            return Err(MatrixCreationError::ZeroDimensions { rows, cols });
        }

        let mut result_matrix = Matrix {
            rows,
            cols,
            data: Vec::new(),
        };

        let size = rows * cols;
        for _ in 0..size {
            result_matrix.data.push(n);
        }

        Ok(result_matrix)
    }

    /// Creates a new Matrix with provided data
    ///
    /// # Arguments
    /// * `rows` - the number of rows
    /// * `cols` - the number of columns
    /// * `data` - the data to be stored in the matrix
    ///
    /// # Returns
    /// A new `Matrix` with defined properties and data.
    ///
    /// # Example
    /// ```
    /// use linears::math::matrix::Matrix;
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let matrix = Matrix::from_vec(2, 2, data);
    /// ```
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, MatrixCreationError> {
        if rows == 0 || cols == 0 {
            return Err(MatrixCreationError::ZeroDimensions { rows, cols });
        }

        if rows * cols != data.len() {
            return Err(MatrixCreationError::DataLengthMismatch {
                rows,
                cols,
                data_len: data.len(),
            });
        }

        Ok(Matrix { rows, cols, data })
    }

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
    /// use linears::math::matrix::Matrix;
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
    /// use linears::math::matrix::Matrix;
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
        let mut result_matrix = Matrix::new_filled(self.rows, other.cols, 0.0).unwrap();

        // matrix[i][j] = matrix[i * cols + j]
        for i in 0..self.rows {
            for j in 0..other.cols {
                result_matrix.data[i * result_matrix.cols + j] = 0.0;
                for k in 0..self.cols {
                    result_matrix.data[i * result_matrix.cols + j] +=
                        self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
            }
        }

        Ok(result_matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_new_filled() {
        // Arrange
        let rows = 2;
        let cols = 2;
        let value = 0.0;

        // Act
        let matrix = Matrix::new_filled(rows, cols, value).unwrap();

        // Assert
        assert_eq!(matrix.data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_matrix_from_vec() {
        // Arrange
        let rows = 2;
        let cols = 2;
        let data = vec![1.0, 2.0, 3.0, 4.0];

        // Act
        let matrix = Matrix::from_vec(rows, cols, data.clone()).unwrap();

        // Assert
        assert_eq!(matrix.rows, rows);
        assert_eq!(matrix.cols, cols);
        assert_eq!(matrix.data, data);
    }

    #[test]
    fn test_matrix_addition() {
        // Arrange
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        // Act
        let result = a.add(&b).unwrap();

        assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_matrix_multiplication() {
        // Arrange
        let a = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Matrix::from_vec(2, 4, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]).unwrap();

        // Act
        let result = a.multiply(&b).unwrap();

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
}
