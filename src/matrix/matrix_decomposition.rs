use crate::{
    matrix::{Matrix, errors::MatrixOperationError},
    utils::{dot, norm},
    vector::Vector,
};

impl Matrix {
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
        let m = self.rows;
        let n = self.cols;

        let mut q = Matrix::new(m, n);
        let mut r = Matrix::new(n, n);

        let mut v_cols: Vec<Vec<f64>> = Vec::with_capacity(n);
        for j in 0..n {
            v_cols.push(self.col(j)?);
        }

        for j in 0..n {
            let mut vj = v_cols[j].clone();

            for i in 0..j {
                let qi = q.col(i)?;
                let rij = dot(&qi, &vj);
                r[(i, j)] = rij;

                for k in 0..m {
                    vj[k] -= rij * qi[k];
                }
            }

            let rjj = norm(&vj);
            if rjj.abs() < 1e-12 {
                return Err(MatrixOperationError::Singular);
            }
            r[(j, j)] = rjj;

            for k in 0..m {
                q[(k, j)] = vj[k] / rjj;
            }
        }

        Ok((q, r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_matrix_lu_decompose() {
        // Arrange
        let m = Matrix::from_vec(3, 3, vec![4.0, 3.0, 2.0, 6.0, 3.0, 0.0, 2.0, 1.0, 1.0]).unwrap();

        // Act
        let (l, u) = m.lu_decompose().unwrap();

        // Assert
        assert_eq!(&l * &u, m);
    }

    #[test]
    fn test_qr_decompose() {
        let a = Matrix::from_vec(
            3,
            3,
            vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
        )
        .unwrap();

        let (q, r) = a.qr_decompose().unwrap();

        // 1. Check Q^T Q = I
        let qtq = &q.transpose() * &q;
        let identity = Matrix::identity(3);
        assert_eq!(qtq, identity);

        // 2. Check R is upper triangular
        for i in 0..r.rows {
            for j in 0..i {
                assert!(r[(i, j)].abs() < 1e-9);
            }
        }

        // 3. Check Q * R = A
        let reconstructed = &q * &r;
        assert_eq!(reconstructed, a);
    }
}
