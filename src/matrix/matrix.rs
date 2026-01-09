use std::ops::{Index, IndexMut};

use crate::{
    matrix::errors::{MatrixCreationError, MatrixOperationError},
    vector::Vector,
};

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

    fn determinant_minor(&self, row_to_remove: usize, col_to_remove: usize) -> Matrix {
        let mut data = Vec::with_capacity((self.rows - 1) * (self.cols - 1));

        for row in 0..self.rows {
            if row == row_to_remove {
                continue;
            }

            for column in 0..self.cols {
                if column == col_to_remove {
                    continue;
                }

                data.push(self[(row, column)])
            }
        }

        Matrix {
            rows: self.rows - 1,
            cols: self.cols - 1,
            data,
        }
    }

    fn determinant_recursive(&self) -> f64 {
        let n = self.rows;

        if n == 1 {
            return self[(0, 0)];
        }

        if n == 2 {
            return self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)];
        }

        let mut determinant = 0.0;

        for col in 0..n {
            let sign = if col % 2 == 0 { 1.0 } else { -1.0 };
            let minor = self.determinant_minor(0, col);
            determinant += sign * self[(0, col)] * minor.determinant_recursive();
        }

        determinant
    }

    /// Calculates the determinant of a matrix
    ///
    /// *Note: This uses recursive Laplace expansion which is
    /// extremely slow for large matrices. This will need
    /// optimising.*
    ///
    /// # Returns
    /// The calculated determinant
    ///
    /// # Example
    /// ```
    /// use linears::matrix::Matrix;
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let matrix = Matrix::from_vec(2, 2, data).unwrap();
    ///
    /// let determinant = matrix.determinant();
    /// ```
    pub fn determinant(&self) -> Result<f64, MatrixOperationError> {
        if !self.is_square() {
            return Err(MatrixOperationError::NotSquare);
        }

        Ok(self.determinant_recursive())
    }

    // TODO: Figure out how this is supposed to work
    pub fn inverse(&self) -> Result<Matrix, MatrixOperationError> {
        if !self.is_square() {
            return Err(MatrixOperationError::NotSquare);
        }

        if self.determinant().expect("already validated to be square") == 0.0 {
            return Err(MatrixOperationError::Singular);
        }

        todo!()
    }

    /// Calculates the LU decomposition of a matrix
    ///
    /// # Returns
    /// Two `Matrix` objects where one is L and one is U.
    ///
    /// # Example
    /// ```
    /// use linears::matrix::Matrix;
    ///
    /// let m = Matrix::from_vec(3, 3, vec![4.0, 3.0, 2.0, 6.0, 3.0, 0.0, 2.0, 1.0, 1.0]).unwrap();
    ///
    /// let (l, u) = m.lu_decompose().unwrap();
    ///
    /// assert_eq!(&l * &u, m);
    /// ```
    pub fn lu_decompose(&self) -> Result<(Matrix, Matrix), MatrixOperationError> {
        let mut l = Matrix::new(self.rows, self.cols);
        let mut u = Matrix::new(self.rows, self.cols);

        // Fill diagonal with 1.0
        for i in 0..self.rows {
            l[(i, i)] = 1.0;
        }

        for i in 0..self.rows {
            // Compute U row i
            for j in i..self.rows {
                let mut sum = 0.0;

                for k in 0..i {
                    sum += l[(i, k)] * u[(k, j)];
                }

                u[(i, j)] = self[(i, j)] - sum;
            }

            // Check for zero pivot
            if u[(i, i)].abs() < 1e-12 {
                return Err(MatrixOperationError::Singular);
            }

            // Compute L column i
            for j in (i + 1)..self.rows {
                let mut sum = 0.0;

                for k in 0..i {
                    sum += l[(j, k)] * u[(k, i)];
                }

                l[(j, i)] = (self[(j, i)] - sum) / u[(i, i)];
            }
        }

        Ok((l, u))
    }

    // TODO: Figure out how this is supposed to work
    pub fn qr_decompose(&self) -> Result<(Matrix, Matrix), MatrixOperationError> {
        todo!()
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
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Act
        let result = m.transpose();

        // Assert
        assert_eq!(result.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
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
    fn test_matrix_scalar_multiply() {
        // Arrange
        let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Act
        let scalar_matrix = m.scalar_multiply(10.0);

        // Assert
        assert_eq!(scalar_matrix.data, vec![10.0, 20.0, 30.0, 40.0]);
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
    fn test_matrix_determinant() {
        // Arrange
        let m =
            Matrix::from_vec(3, 3, vec![2.0, 5.0, 3.0, 1.0, -2.0, -1.0, 1.0, 3.0, 4.0]).unwrap();

        // Act
        let determinant = m.determinant().unwrap();

        // Assert
        assert_eq!(determinant, -20.0);
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
    fn test_matrix_map_iterator() {
        // Arrange
        let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Act
        let result = m.map(|value| value * value);

        // Assert
        assert_eq!(result.data, vec![1.0, 4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_matrix_lu_decompose() {
        // Arrange
        let m = Matrix::from_vec(3, 3, vec![4.0, 3.0, 2.0, 6.0, 3.0, 0.0, 2.0, 1.0, 1.0]).unwrap();

        // Act
        let (l, u) = m.lu_decompose().unwrap();

        // Assert
        assert_eq!(&l * &u, m);
    }
}
