use crate::dataframe::{Column, Dtype, column::IntoColumn};

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
}
