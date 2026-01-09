use crate::math::errors::{MatrixCreationError, MatrixOperationError};
use std::ops::{Index, IndexMut};

// Row-major implementation of a matrix
#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    #[inline]
    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

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

        Ok(Matrix {
            rows,
            cols,
            data: vec![n; rows * cols],
        })
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
    /// let matrix = Matrix::from_vec(2, 2, data).unwrap();
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

    /// Gets the value from a matrix at index
    ///
    /// # Arguments
    /// * `row` - index of the row
    /// * `col` - index of the column
    ///
    /// # Returns
    /// The value stored at that index
    ///
    /// # Example
    /// ```
    /// use linears::math::matrix::Matrix;
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let matrix = Matrix::from_vec(2, 2, data).unwrap();
    ///
    /// let value = matrix.get(1, 1).unwrap();
    /// ```
    pub fn get(&self, row: usize, col: usize) -> Result<f64, MatrixOperationError> {
        if row >= self.rows || col >= self.cols {
            return Err(MatrixOperationError::OutOfBounds {
                index_row: row,
                index_col: col,
                actual_row: self.rows,
                actual_col: self.cols,
            });
        }

        Ok(self[(row, col)])
    }

    /// Sets the value in a matrix at index
    ///
    /// # Arguments
    /// * `row` - index of the row
    /// * `col` - index of the column
    /// * `value` - the value to set
    ///
    /// # Example
    /// ```
    /// use linears::math::matrix::Matrix;
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let mut matrix = Matrix::from_vec(2, 2, data).unwrap();
    ///
    /// matrix.set(1, 1, 5.0).unwrap();
    /// ```
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<(), MatrixOperationError> {
        if row >= self.rows || col >= self.cols {
            return Err(MatrixOperationError::OutOfBounds {
                index_row: row,
                index_col: col,
                actual_row: self.rows,
                actual_col: self.cols,
            });
        }

        self[(row, col)] = value;
        Ok(())
    }

    // TODO: Implement these methods
    // // Row/column extraction
    // pub fn row(&self, index: usize) -> Result<Vec<f64>, MatrixOperationError>;
    // pub fn col(&self, index: usize) -> Result<Vec<f64>, MatrixOperationError>;
    //
    // // Functional transform
    // pub fn map<F>(&self, f: F) -> Matrix
    // where
    //     F: Fn(f64) -> f64;
    //
    // // Shape utilities
    // pub fn is_square(&self) -> bool;
    //
    // // Scalar operations
    // pub fn scalar_mul(&self, scalar: f64) -> Matrix;
    // pub fn scalar_add(&self, scalar: f64) -> Matrix;
    //
    // // Determinant & inverse
    // pub fn determinant(&self) -> Result<f64, MatrixOperationError>;
    // pub fn inverse(&self) -> Result<Matrix, MatrixOperationError>;
    //
    // // Decompositions
    // pub fn lu_decompose(&self) -> Result<(Matrix, Matrix), MatrixOperationError>;
    // pub fn qr_decompose(&self) -> Result<(Matrix, Matrix), MatrixOperationError>;
    //
    // // Matrix–vector multiplication
    // pub fn mul_vector(&self, vec: &Vector) -> Result<Vector, MatrixOperationError>;

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
        if self.cols != other.rows {
            return Err(MatrixOperationError::InvalidDimensions {
                a_rows: self.rows,
                a_cols: self.cols,
                b_rows: other.rows,
                b_cols: other.cols,
            });
        }

        let mut result_matrix =
            Matrix::new_filled(self.rows, other.cols, 0.0).expect("dimensions already validated");

        // matrix[i][j] = matrix[i * cols + j]
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;

                for k in 0..self.cols {
                    sum += self.data[self.idx(i, k)] * other.data[other.idx(k, j)];
                }

                let idx = result_matrix.idx(i, j);
                result_matrix.data[idx] = sum;
            }
        }

        Ok(result_matrix)
    }

    /// Transpose the matrix
    ///
    /// # Returns
    /// A new `Matrix` with transposed value.
    ///
    /// # Example
    /// ```
    /// use linears::math::matrix::Matrix;
    /// let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    ///
    /// let result = a.transpose().unwrap();
    ///
    /// assert_eq!(result.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    /// ```
    pub fn transpose(&self) -> Result<Matrix, MatrixOperationError> {
        let mut result_matrix =
            Matrix::new_filled(self.cols, self.rows, 0.0).expect("dimensions already validated");

        for i in 0..self.rows {
            for j in 0..self.cols {
                let idx = result_matrix.idx(j, i);
                result_matrix.data[idx] = self.data[self.idx(i, j)];
            }
        }

        Ok(result_matrix)
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        &self.data[self.idx(row, col)]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        let idx = self.idx(row, col);
        &mut self.data[idx]
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }

        // f64 float tolerance
        const EPS: f64 = 1e-9;

        self.data
            .iter()
            .zip(&other.data)
            .all(|(a, b)| (a - b).abs() < EPS)
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
    fn test_matrix_getter() {
        // Arrange
        let matrix = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Act
        let value = matrix.get(1, 1).unwrap();

        // Assert
        assert_eq!(value, 4.0);
    }

    #[test]
    fn test_matrix_setter() {
        // Arrange
        let mut matrix = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Act
        matrix.set(1, 1, 10.0).unwrap();

        // Assert
        assert_eq!(matrix[(1, 1)], 10.0);
    }

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
    fn test_matrix_transposition() {
        // Arrange
        let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Act
        let result = a.transpose().unwrap();

        // Assert
        assert_eq!(result.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
