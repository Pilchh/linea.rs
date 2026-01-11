use std::fmt::{self, Display};

use crate::dataframe::{Column, Dtype, column::IntoColumn, scalar::Scalar};

#[derive(Debug, Clone)]
pub struct Series {
    pub name: String,
    pub column: Column,
}

impl Series {
    pub fn new(name: String, data: Column) -> Series {
        Series { name, column: data }
    }

    pub fn from_vec<T: IntoColumn>(name: String, data: Vec<T>) -> Series {
        Series {
            name,
            column: T::into_column(data),
        }
    }

    pub fn from_dtype(name: String, dtype: Dtype) -> Series {
        let column = match dtype {
            Dtype::Int64 => Column::Int64(Vec::new()),
            Dtype::Float64 => Column::Int64(Vec::new()),
            Dtype::String => Column::Int64(Vec::new()),
            Dtype::Bool => Column::Int64(Vec::new()),
        };

        Series { name, column }
    }

    pub fn from_column(name: String, column: Column) -> Series {
        Series { name, column }
    }

    pub fn len(&self) -> usize {
        self.column.len()
    }

    pub fn eq(&self, value: impl Into<Scalar>) -> Series {
        Series {
            name: self.name.clone(),
            column: self.column.eq(value),
        }
    }
}

impl fmt::Display for Series {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Series: {{")?;
        writeln!(f, "  name: {}", self.name)?;
        writeln!(f, "  column: {{")?;

        for line in self.column.as_strings() {
            writeln!(f, "    {}", line)?;
        }

        writeln!(f, "  }}")?;
        writeln!(f, "}}")?;

        Ok(())
    }
}
