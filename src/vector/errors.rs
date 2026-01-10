use thiserror::Error;

#[derive(Debug, Error)]
pub enum VectorOperationError {
    #[error("index is out of bounds (tried to get index {index} on size {size})")]
    OutOfBounds { index: usize, size: usize },
}
