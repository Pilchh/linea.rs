use std::{fs, path::Path};

use crate::dataframe::{DataFrame, Dtype, errors::IOError};

fn infer_dtype(column: &[&str]) -> Dtype {
    if column.iter().all(|s| s.parse::<i64>().is_ok()) {
        Dtype::Int64
    } else if column.iter().all(|s| s.parse::<f64>().is_ok()) {
        Dtype::Float64
    } else if column.iter().all(|s| s == &"true" || s == &"false") {
        Dtype::Bool
    } else {
        Dtype::String
    }
}

impl DataFrame {
    pub fn from_csv(csv: String) -> Result<DataFrame, IOError> {
        let path = Path::new(&csv);
        if !path.exists() {
            return Err(IOError::FileNotFound { path: csv });
        }

        let contents = fs::read_to_string(path).expect("should have been able to read file");
        let rows: Vec<&str> = contents.lines().collect();

        if rows.len() <= 1 {
            return Err(IOError::FileTooSmall);
        }

        let headers = rows[0].split(',').collect::<Vec<&str>>();
        let data = rows[1..].to_vec();

        let mut columns: Vec<Vec<&str>> = vec![Vec::new(); headers.len()];

        for row in &data {
            let cells: Vec<&str> = row.split(',').collect();

            if cells.len() != headers.len() {
                return Err(IOError::RowLengthMismatch);
            }

            for (i, cell) in cells.iter().enumerate() {
                columns[i].push(*cell);
            }
        }

        let mut df = DataFrame::new();

        for (i, column) in columns.iter().enumerate() {
            let dtype = infer_dtype(column);
            let name = headers[i].to_string();

            match dtype {
                Dtype::Int64 => {
                    let values: Vec<i64> =
                        column.iter().map(|s| s.parse::<i64>().unwrap()).collect();

                    df.column(&name, values);
                }
                Dtype::Float64 => {
                    let values: Vec<f64> =
                        column.iter().map(|s| s.parse::<f64>().unwrap()).collect();

                    df.column(&name, values);
                }
                Dtype::String => {
                    let values: Vec<String> = column.iter().map(|s| s.to_string()).collect();

                    df.column(&name, values);
                }
                Dtype::Bool => {
                    let values: Vec<bool> = column.iter().map(|s| *s == "true").collect();

                    df.column(&name, values);
                }
            }
        }

        Ok(df)
    }
}
