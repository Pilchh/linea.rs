use std::ops::{Index, IndexMut};

pub struct Vector {
    pub size: usize,
    pub data: Vec<f64>,
}

impl Vector {
    /// Creates a new Vector with provided data
    ///
    /// # Arguments
    /// * `data` - the data to be stored in the vector
    ///
    /// # Returns
    /// A new `Vector` with defined data.
    ///
    /// # Example
    /// ```
    /// use linears::vector::Vector;
    ///
    /// let data = vec![1.0, 2.0, 3.0];
    /// let vector = Vector::from_vec(data);
    /// ```
    pub fn from_vec(data: Vec<f64>) -> Vector {
        Vector {
            size: data.len(),
            data,
        }
    }

    //     pub fn get(&self, index: usize) -> Result<f64, VectorOperationError>;
    //     pub fn set(&mut self, index: usize, value: f64) -> Result<(), VectorOperationError>;
    //
    //     pub fn add(&self, other: &Vector) -> Result<Vector, VectorOperationError>;
    //     pub fn sub(&self, other: &Vector) -> Result<Vector, VectorOperationError>;
    //     pub fn scalar_mul(&self, scalar: f64) -> Vector;
    //     pub fn dot(&self, other: &Vector) -> Result<f64, VectorOperationError>;
}

impl Index<usize> for Vector {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_from_vec() {
        // Arrange
        let data = vec![1.0, 2.0, 3.0];

        // Act
        let v = Vector::from_vec(data);

        // Assert
        assert_eq!(v.data, vec![1.0, 2.0, 3.0]);
    }
}
