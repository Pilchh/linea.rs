use thiserror::Error;

#[derive(Debug, Error)]
pub enum DataFrameError {
    #[error(
        "series length does not match dataframe length (series length {s_len}, dataframe length {df_len})"
    )]
    SeriesLengthMismatch { s_len: usize, df_len: usize },
    #[error(
        "column name length does not match column dtypes (name length {n_len}, dtype length {d_len})"
    )]
    ColumnDtypeMismatch { n_len: usize, d_len: usize },
}

#[derive(Debug, Error)]
pub enum IOError {
    #[error("requested file could not be found (${path})")]
    FileNotFound { path: String },
    #[error("file size is too small, 2 lines minimum to form a dataframe")]
    FileTooSmall,
    #[error("length of rows within file do not match")]
    RowLengthMismatch,
}
