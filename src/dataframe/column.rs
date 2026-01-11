use std::fmt;

use crate::dataframe::{dtype::Dtype, scalar::Scalar};

#[derive(Debug, Clone)]
pub enum Column {
    Int64(Vec<i64>),
    Float64(Vec<f64>),
    String(Vec<String>),
    Bool(Vec<bool>),
}

impl Column {
    pub fn as_strings(&self) -> Vec<String> {
        match self {
            Column::Int64(v) => v.iter().map(|x| x.to_string()).collect(),
            Column::Float64(v) => v.iter().map(|x| x.to_string()).collect(),
            Column::String(v) => v.clone(),
            Column::Bool(v) => v.iter().map(|x| x.to_string()).collect(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Column::Int64(v) => v.len(),
            Column::Float64(v) => v.len(),
            Column::String(v) => v.len(),
            Column::Bool(v) => v.len(),
        }
    }

    pub fn dtype(&self) -> Dtype {
        match self {
            Column::Int64(_) => Dtype::Int64,
            Column::Float64(_) => Dtype::Float64,
            Column::String(_) => Dtype::String,
            Column::Bool(_) => Dtype::Bool,
        }
    }

    pub fn head(&self, n: usize) -> Column {
        let n = n.min(self.len());

        match self {
            Column::Int64(v) => Column::Int64(v[..n].to_vec()),
            Column::Float64(v) => Column::Float64(v[..n].to_vec()),
            Column::String(v) => Column::String(v[..n].to_vec()),
            Column::Bool(v) => Column::Bool(v[..n].to_vec()),
        }
    }

    pub fn eq(&self, value: impl Into<Scalar>) -> Column {
        let value: Scalar = value.into();

        match (self, value) {
            (Column::Int64(col), Scalar::Int64(v)) => {
                Column::Bool(col.iter().map(|x| *x == v).collect())
            }
            (Column::Float64(col), Scalar::Float64(v)) => {
                Column::Bool(col.iter().map(|x| *x == v).collect())
            }
            (Column::String(col), Scalar::String(v)) => {
                Column::Bool(col.iter().map(|x| *x == v).collect())
            }
            (Column::Bool(col), Scalar::Bool(v)) => {
                Column::Bool(col.iter().map(|x| *x == v).collect())
            }

            _ => panic!("dtype mismatch"),
        }
    }

    pub fn cast(&self, new_type: &Dtype) -> Self {
        match (self, new_type) {
            (Column::Int64(col), Dtype::Float64) => {
                Column::Float64(col.iter().map(|v| *v as f64).collect())
            }
            _ => {
                todo!()
            }
        }
    }
}

pub trait IntoColumn {
    fn into_column(data: Vec<Self>) -> Column
    where
        Self: Sized;
}

impl IntoColumn for i64 {
    fn into_column(data: Vec<Self>) -> Column {
        Column::Int64(data)
    }
}

impl IntoColumn for f64 {
    fn into_column(data: Vec<Self>) -> Column {
        Column::Float64(data)
    }
}

impl IntoColumn for String {
    fn into_column(data: Vec<Self>) -> Column {
        Column::String(data)
    }
}

impl IntoColumn for &str {
    fn into_column(data: Vec<Self>) -> Column {
        Column::String(data.into_iter().map(|s| s.to_string()).collect())
    }
}

impl IntoColumn for bool {
    fn into_column(data: Vec<Self>) -> Column {
        Column::Bool(data)
    }
}

impl fmt::Display for Column {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.len() == 0 {
            writeln!(f, "Empty Column")?;
            return Ok(());
        }

        match self {
            Column::Int64(v) => {
                for x in v {
                    writeln!(f, "{x}")?;
                }
            }
            Column::Float64(v) => {
                for x in v {
                    writeln!(f, "{x}, ")?;
                }
            }
            Column::String(v) => {
                for x in v {
                    writeln!(f, "{x}, ")?;
                }
            }
            Column::Bool(v) => {
                for x in v {
                    writeln!(f, "{x}, ")?;
                }
            }
        }

        Ok(())
    }
}
