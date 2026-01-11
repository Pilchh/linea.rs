use std::fmt::{self, Display};

use crate::dataframe::{Column, Dtype, column::IntoColumn, scalar::Scalar};

#[derive(Debug, Clone)]
pub struct Series {
    pub name: String,
    pub column: Column,
    pub dtype: Dtype,
}

impl Series {
    pub fn new(name: String, data: Column) -> Series {
        Series {
            name,
            column: data.clone(),
            dtype: data.dtype(),
        }
    }

    pub fn from_vec<T: IntoColumn>(name: String, data: Vec<T>) -> Series {
        let column = T::into_column(data);

        Series {
            name,
            column: column.clone(),
            dtype: column.dtype(),
        }
    }

    pub fn from_dtype(name: String, dtype: Dtype) -> Series {
        let column = match dtype {
            Dtype::Int64 => Column::Int64(Vec::new()),
            Dtype::Float64 => Column::Int64(Vec::new()),
            Dtype::String => Column::Int64(Vec::new()),
            Dtype::Bool => Column::Int64(Vec::new()),
        };

        Series {
            name,
            column,
            dtype,
        }
    }

    pub fn from_column(name: String, column: Column) -> Series {
        Series {
            name,
            column: column.clone(),
            dtype: column.dtype(),
        }
    }

    pub fn len(&self) -> usize {
        self.column.len()
    }

    pub fn eq(&self, value: impl Into<Scalar>) -> Series {
        Series {
            name: self.name.clone(),
            column: self.column.clone().eq(value),
            dtype: Dtype::Bool,
        }
    }

    pub fn filter(&self, mask: &Series) -> Series {
        if let Column::Bool(mask) = &mask.column {
            let filtered_column = match &self.column {
                Column::Int64(v) => Column::Int64(
                    v.iter()
                        .zip(mask)
                        .filter_map(|(x, &m)| if m { Some(*x) } else { None })
                        .collect(),
                ),
                Column::Float64(v) => Column::Float64(
                    v.iter()
                        .zip(mask)
                        .filter_map(|(x, &m)| if m { Some(*x) } else { None })
                        .collect(),
                ),
                Column::String(v) => Column::String(
                    v.iter()
                        .zip(mask)
                        .filter_map(|(x, &m)| if m { Some(x.clone()) } else { None })
                        .collect(),
                ),
                Column::Bool(v) => Column::Bool(
                    v.iter()
                        .zip(mask)
                        .filter_map(|(x, &m)| if m { Some(*x) } else { None })
                        .collect(),
                ),
            };

            Series {
                name: self.name.clone(),
                column: filtered_column,
                dtype: self.dtype.clone(),
            }
        } else {
            panic!("mask is not a boolean column");
        }
    }
}

impl fmt::Display for Series {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Series: {{")?;
        writeln!(f, "  name: {}", self.name)?;
        writeln!(f, "  dtype: {}", self.dtype)?;
        writeln!(f, "  column: {{")?;

        for line in self.column.as_strings() {
            writeln!(f, "    {}", line)?;
        }

        writeln!(f, "  }}")?;
        writeln!(f, "}}")?;

        Ok(())
    }
}
