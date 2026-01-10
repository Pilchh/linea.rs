use thiserror::Error;

#[derive(Debug, Error)]
pub enum VectorOperationError {
    #[error("index is out of bounds (tried to get index {index} on size {size})")]
    OutOfBounds { index: usize, size: usize },
    #[error("the dimensions of the two vectors are not the same (got {a_size} and {b_size})")]
    InvalidDimensions { a_size: usize, b_size: usize },
}
