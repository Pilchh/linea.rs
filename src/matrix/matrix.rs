use std::ops::{Index, IndexMut};

use crate::matrix::errors::{MatrixCreationError, MatrixOperationError};

// Row-major implementation of a matrix
#[derive(Debug)]
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

    /// Creates a new Matrix with defined size filled with all 0.0
    ///
    /// # Arguments
    /// * `rows` - the number of rows
    /// * `cols` - the number of columns
    ///
    /// # Returns
    /// A new `Matrix` with defined properties filled with zeros.
    ///
    /// # Example
    /// ```
    /// use linears::matrix::Matrix;
    ///
    /// let matrix = Matrix::new(2, 2);
    /// ```
    pub fn new(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
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
    /// use linears::matrix::Matrix;
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

    // TODO: Add docstring
    pub fn identity(n: usize) -> Matrix {
        let mut m = Matrix::new(n, n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }

    /// Fills a matrix with a value
    ///
    /// # Arguments
    /// * `n` - the value to fill the matrix with
    ///
    /// # Example
    /// ```
    /// use linears::matrix::Matrix;
    ///
    /// let mut matrix = Matrix::new(2, 2);
    /// matrix.fill(0.0);
    /// ```
    pub fn fill(&mut self, n: f64) -> &mut Matrix {
        self.data.fill(n);
        self
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
    /// use linears::matrix::Matrix;
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let matrix = Matrix::from_vec(2, 2, data).unwrap();
    ///
    /// let value = matrix.get(1, 1).unwrap();
    /// ```
    pub fn get(&self, row: usize, col: usize) -> Result<f64, MatrixOperationError> {
        if row >= self.rows || col >= self.cols {
            return Err(MatrixOperationError::OutOfBoundsMat {
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
    /// use linears::matrix::Matrix;
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let mut matrix = Matrix::from_vec(2, 2, data).unwrap();
    ///
    /// matrix.set(1, 1, 5.0).unwrap();
    /// ```
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<(), MatrixOperationError> {
        if row >= self.rows || col >= self.cols {
            return Err(MatrixOperationError::OutOfBoundsMat {
                index_row: row,
                index_col: col,
                actual_row: self.rows,
                actual_col: self.cols,
            });
        }

        self[(row, col)] = value;
        Ok(())
    }

    /// Gets an entire row from a matrix
    ///
    /// # Arguments
    /// * `index` - index of the row
    ///
    /// # Returns
    /// A vector containing all the values in the row
    ///
    /// # Example
    /// ```
    /// use linears::matrix::Matrix;
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let mut matrix = Matrix::from_vec(2, 2, data).unwrap();
    ///
    /// let row = matrix.row(0).unwrap();
    /// ```
    pub fn row(&self, index: usize) -> Result<Vec<f64>, MatrixOperationError> {
        if index >= self.rows {
            return Err(MatrixOperationError::OutOfBoundsRow {
                index_row: index,
                actual_row: self.rows,
            });
        }

        let start = index * self.cols;
        Ok(self.data[start..start + self.cols].to_vec())
    }

    /// Gets an entire column from a matrix
    ///
    /// # Arguments
    /// * `index` - index of the column
    ///
    /// # Returns
    /// A vector containing all the values in the column
    ///
    /// # Example
    /// ```
    /// use linears::matrix::Matrix;
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let mut matrix = Matrix::from_vec(2, 2, data).unwrap();
    ///
    /// let column = matrix.col(0).unwrap();
    /// ```
    pub fn col(&self, index: usize) -> Result<Vec<f64>, MatrixOperationError> {
        if index >= self.cols {
            return Err(MatrixOperationError::OutOfBoundsCol {
                index_col: index,
                actual_col: self.cols,
            });
        }

        let mut output_vec: Vec<f64> = Vec::with_capacity(self.rows);

        for i in 0..self.rows {
            output_vec.push(self[(i, index)]);
        }

        Ok(output_vec)
    }

    /// Checks if a matrix is square
    ///
    /// # Returns
    /// If the matrix is square
    ///
    /// # Example
    /// ```
    /// use linears::matrix::Matrix;
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let mut matrix = Matrix::from_vec(2, 2, data).unwrap();
    ///
    /// let is_square = matrix.is_square();
    /// ```
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Applies a function to every element of the matrix and returns a new matrix.
    ///
    /// The dimensions and layout of the matrix is preserved.
    ///
    /// # Arguments
    /// * `f` - A function that takes a single `f64` and returns a transformed `f64`
    ///
    /// # Returns
    /// A new `Matrix` with the same dimensions, where each element has been
    /// transformed by the function.
    ///
    /// # Example
    /// ```
    /// use linears::matrix::Matrix;
    ///
    /// let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let squared = m.map(|value| value * value);
    ///
    /// assert_eq!(squared.data, vec![1.0, 4.0, 9.0, 16.0]);
    /// ```
    pub fn map<F>(&self, f: F) -> Matrix
    where
        F: Fn(f64) -> f64,
    {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().copied().map(f).collect(),
        }
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
        let mut m = Matrix::new(rows, cols);
        m.fill(value);

        // Assert
        assert_eq!(m.data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_matrix_fill() {
        // Arrange
        let mut m = Matrix::new(2, 2);

        // Act
        m.fill(0.0);

        // Assert
        assert_eq!(m.data, vec![0.0; 4]);
    }

    #[test]
    fn test_matrix_getter() {
        // Arrange
        let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Act
        let value = m.get(1, 1).unwrap();

        // Assert
        assert_eq!(value, 4.0);
    }

    #[test]
    fn test_matrix_setter() {
        // Arrange
        let mut m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Act
        m.set(1, 1, 10.0).unwrap();

        // Assert
        assert_eq!(m[(1, 1)], 10.0);
    }

    #[test]
    fn test_matrix_from_vec() {
        // Arrange
        let rows = 2;
        let cols = 2;
        let data = vec![1.0, 2.0, 3.0, 4.0];

        // Act
        let m = Matrix::from_vec(rows, cols, data.clone()).unwrap();

        // Assert
        assert_eq!(m.rows, rows);
        assert_eq!(m.cols, cols);
        assert_eq!(m.data, data);
    }

    #[test]
    fn test_matrix_row() {
        // Arrange
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Act
        let row = m.row(0).unwrap();

        // Assert
        assert_eq!(row, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_matrix_col() {
        // Arrange
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Act
        let row = m.col(0).unwrap();

        // Assert
        assert_eq!(row, vec![1.0, 4.0]);
    }

    #[test]
    fn test_matrix_is_square() {
        // Arrange
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Act
        let is_a_square = a.is_square();
        let is_b_square = b.is_square();

        // Assert
        assert_eq!(is_a_square, true);
        assert_eq!(is_b_square, false);
    }

    #[test]
    fn test_matrix_map_iterator() {
        // Arrange
        let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Act
        let result = m.map(|value| value * value);

        // Assert
        assert_eq!(result.data, vec![1.0, 4.0, 9.0, 16.0]);
    }
}
