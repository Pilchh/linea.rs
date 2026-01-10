use std::ops::{Index, IndexMut};

use crate::vector::errors::VectorOperationError;

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

    /// Gets the value from a vector at index
    ///
    /// # Arguments
    /// * `index` - index of the data
    ///
    /// # Returns
    /// The value stored at that index
    ///
    /// # Example
    /// ```
    /// use linears::vector::Vector;
    ///
    /// let vector = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    ///
    /// let value = vector.get(1).unwrap();
    ///
    /// assert_eq!(value, 2.0);
    /// ```
    pub fn get(&self, index: usize) -> Result<f64, VectorOperationError> {
        if index >= self.size {
            return Err(VectorOperationError::OutOfBounds {
                index: index,
                size: self.size,
            });
        }

        Ok(self.data[index])
    }

    /// Sets the value at an index of a vector
    ///
    /// # Arguments
    /// * `index` - index of the data
    /// * `value` - the value to set
    ///
    /// # Example
    /// ```
    /// use linears::vector::Vector;
    ///
    /// let mut vector = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    ///
    /// vector.set(1, 20.0).unwrap();
    ///
    /// assert_eq!(vector.get(1).unwrap(), 20.0);
    /// ```
    pub fn set(&mut self, index: usize, value: f64) -> Result<(), VectorOperationError> {
        if index >= self.size {
            return Err(VectorOperationError::OutOfBounds {
                index: index,
                size: self.size,
            });
        }

        self.data[index] = value;
        Ok(())
    }

    /// Adds two vectors together
    ///
    /// # Arguments
    /// * `other` - the vector to add
    ///
    /// # Returns
    /// A new `Vector` with the computed value.
    ///
    /// # Example
    /// ```
    /// use linears::vector::Vector;
    /// let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = Vector::from_vec(vec![5.0, 6.0, 7.0]);
    ///
    /// let result = a.add(&b).unwrap();
    ///  
    /// assert_eq!(result.data, vec![6.0, 8.0, 10.0]);
    /// ```
    pub fn add(&self, other: &Vector) -> Result<Vector, VectorOperationError> {
        if self.size != other.size {
            return Err(VectorOperationError::InvalidDimensions {
                a_size: self.size,
                b_size: other.size,
            });
        }

        let mut result_vector = Vector {
            size: self.size,
            data: Vec::with_capacity(self.size),
        };

        for i in 0..self.size {
            result_vector.data.push(self.data[i] + other.data[i]);
        }

        Ok(result_vector)
    }

    /// Substracts one vector from another
    ///
    /// # Arguments
    /// * `other` - the vector to subtract
    ///
    /// # Returns
    /// A new `Vector` with the computed value.
    ///
    /// # Example
    /// ```
    /// use linears::vector::Vector;
    /// let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = Vector::from_vec(vec![5.0, 6.0, 7.0]);
    ///
    /// let result = a.sub(&b).unwrap();
    ///  
    /// assert_eq!(result.data, vec![-4.0, -4.0, -4.0]);
    /// ```
    pub fn sub(&self, other: &Vector) -> Result<Vector, VectorOperationError> {
        if self.size != other.size {
            return Err(VectorOperationError::InvalidDimensions {
                a_size: self.size,
                b_size: other.size,
            });
        }

        let mut result_vector = Vector {
            size: self.size,
            data: Vec::with_capacity(self.size),
        };

        for i in 0..self.size {
            result_vector.data.push(self.data[i] - other.data[i]);
        }

        Ok(result_vector)
    }

    /// Multiplies a vector with a scalar
    ///
    /// # Arguments
    /// * `scalar` - the scalar to multiply by
    ///
    /// # Returns
    /// A new `Vector` with the computed value.
    ///
    /// # Example
    /// ```
    /// use linears::vector::Vector;
    /// let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    ///
    /// let result = v.scalar_mul(10.0);
    ///  
    /// assert_eq!(result.data, vec![10.0, 20.0, 30.0]);
    /// ```
    pub fn scalar_mul(&self, scalar: f64) -> Vector {
        let mut result_vector = Vector {
            size: self.size,
            data: Vec::with_capacity(self.size),
        };

        for i in 0..self.size {
            result_vector.data.push(self.data[i] * scalar);
        }

        result_vector
    }

    /// Calculates to dot product of two vectors
    ///
    /// # Arguments
    /// * `other` - the second vector
    ///
    /// # Returns
    /// A new `Vector` with the computed value.
    ///
    /// # Example
    /// ```
    /// use linears::vector::Vector;
    /// let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = Vector::from_vec(vec![4.0, 5.0, 6.0]);
    ///
    /// let result = a.dot(&b).unwrap();
    ///  
    /// assert_eq!(result, 32.0);
    /// ```
    pub fn dot(&self, other: &Vector) -> Result<f64, VectorOperationError> {
        if self.size != other.size {
            return Err(VectorOperationError::InvalidDimensions {
                a_size: self.size,
                b_size: other.size,
            });
        }

        let mut result = 0.0;

        for i in 0..self.size {
            result += self.data[i] * other.data[i];
        }

        Ok(result)
    }
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

impl std::ops::Add for &Vector {
    type Output = Vector;

    fn add(self, other: Self) -> Vector {
        self.add(other).unwrap()
    }
}

impl std::ops::Sub for &Vector {
    type Output = Vector;

    fn sub(self, other: Self) -> Vector {
        self.sub(other).unwrap()
    }
}

impl std::ops::Mul<f64> for &Vector {
    type Output = Vector;

    fn mul(self, scalar: f64) -> Vector {
        self.scalar_mul(scalar)
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

    #[test]
    fn test_vector_get() {
        // Arrange
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);

        // Act
        let result = v.get(1).unwrap();

        // Assert
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_vector_set() {
        // Arrange
        let mut v = Vector::from_vec(vec![1.0, 2.0, 3.0]);

        // Act
        v.set(1, 20.0).unwrap();

        // Assert
        assert_eq!(v.data[1], 20.0);
    }

    #[test]
    fn test_vector_add() {
        // Arrange
        let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Vector::from_vec(vec![4.0, 5.0, 6.0]);

        // Act
        let result = &a + &b;

        // Assert
        assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vector_sub() {
        // Arrange
        let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Vector::from_vec(vec![4.0, 5.0, 6.0]);

        // Act
        let result = &a - &b;

        // Assert
        assert_eq!(result.data, vec![-3.0, -3.0, -3.0]);
    }

    #[test]
    fn test_vector_dot() {
        // Arrange
        let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Vector::from_vec(vec![4.0, 5.0, 6.0]);

        // Act
        let result = a.dot(&b).unwrap();

        // Assert
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_vector_scalar_mul() {
        // Arrange
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);

        // Act
        let result = &v * 10.0;

        // Assert
        assert_eq!(result.data, vec![10.0, 20.0, 30.0]);
    }
}
