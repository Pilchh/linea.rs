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

#[derive(Debug, Error)]
pub enum MatrixOperationError {
    #[error(
        "the matrices have invalid dimensions for this operation ({a_rows}, {a_cols} and {b_rows}, {b_cols})"
    )]
    InvalidDimensions {
        a_rows: usize,
        a_cols: usize,
        b_rows: usize,
        b_cols: usize,
    },
}
