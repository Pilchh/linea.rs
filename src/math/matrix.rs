use std::error::Error;

// Row-major implementation of a matrix
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn add(&self, other: &Matrix) -> Matrix {
        todo!();
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, String> {
        todo!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_addition() {
        // Arrange
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        // Act
        let result = a.add(&b);

        assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
    }
}
