use thiserror::Error;

#[derive(Debug, Error)]
pub enum MatrixCreationError {
    #[error("matrix dimensions must be greater than zero (got {rows} x {cols})")]
    ZeroDimensions { rows: usize, cols: usize },

    #[error("data length {data_len} does not match matrix size {rows} x {cols}")]
    DataLengthMismatch {
        rows: usize,
        cols: usize,
        data_len: usize,
    },
}
