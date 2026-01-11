use std::fmt;

use crate::dataframe::{Column, Series, column::IntoColumn, errors::DataFrameError};

pub struct DataFrame {
    series: Vec<Series>,
    nrows: usize,
}

impl DataFrame {
    pub fn new() -> DataFrame {
        DataFrame {
            series: Vec::new(),
            nrows: 0,
        }
    }

    pub fn empty() -> DataFrame {
        DataFrame {
            series: Vec::new(),
            nrows: 0,
        }
    }

    pub fn column<T: IntoColumn>(&mut self, name: &str, values: Vec<T>) -> &mut Self {
        if self.nrows != 0 {
            if values.len() != self.nrows {
                panic!("Series length mismatch: {} vs {}", values.len(), self.nrows);
            }
        }

        if self.nrows == 0 {
            self.nrows = values.len();
        }

        let series = Series::from_vec(name.into(), values);
        self.add_series(series);

        self
    }

    pub fn insert(&mut self, index: usize, value: Series) -> Result<(), DataFrameError> {
        if self.nrows != 0 {
            if value.len() != self.nrows {
                return Err(DataFrameError::SeriesLengthMismatch {
                    s_len: value.len(),
                    df_len: self.nrows,
                });
            }
        }

        if self.nrows == 0 {
            self.nrows = value.len();
        }

        self.insert_series(index, value);
        Ok(())
    }

    pub fn remove(&mut self, name: &str) -> &mut Self {
        for i in 0..self.series.len() - 1 {
            if &self.series[i].name == name {
                self.remove_series(i);
                break;
            }
        }

        self
    }

    pub fn head(&self, n: usize) -> DataFrame {
        let mut df = DataFrame::new();

        for series in &self.series {
            let column = &series.column;
            let head = column.head(n);

            let new_series = Series::from_column(series.name.clone(), head);
            df.add_series(new_series);
        }

        df
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.series.len(), self.nrows)
    }

    fn assert_compatible_series(&mut self, series: &Series) {
        if self.nrows != 0 {
            if self.nrows != series.len() {
                panic!(
                    "series size mismatch: df has {} rows, series has {}",
                    self.nrows,
                    series.len()
                );
            }
        }

        if self.nrows == 0 {
            self.nrows = series.len();
        }
    }

    fn remove_series(&mut self, index: usize) {
        self.series.remove(index);

        if self.series.is_empty() {
            self.nrows = 0;
        }
    }

    fn add_series(&mut self, series: Series) {
        self.assert_compatible_series(&series);
        self.series.push(series);
    }

    fn insert_series(&mut self, index: usize, series: Series) {
        self.assert_compatible_series(&series);
        self.series.insert(index, series);
    }
}

const PADDING: usize = 15;

impl fmt::Display for DataFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.series.len() == 0 {
            writeln!(f, "Empty Dataframe")?;
            return Ok(());
        }

        writeln!(f, "{}", "-".repeat(&self.series.len() * PADDING))?;

        for s in &self.series {
            write!(f, "{:<p$}", &s.name, p = PADDING)?
        }
        writeln!(f, "\n{}", "-".repeat(&self.series.len() * PADDING))?;

        for i in 0..self.nrows {
            for s in &self.series {
                match &s.column {
                    Column::String(v) => {
                        let s = format!("\"{}\"", v[i]);
                        write!(f, "{:<p$}", s, p = PADDING)?;
                    }
                    Column::Int64(v) => {
                        write!(f, "{:<p$}", v[i], p = PADDING)?;
                    }
                    Column::Float64(v) => {
                        let s = format!("{:.1}", v[i]);
                        write!(f, "{:<p$}", s, p = PADDING)?;
                    }
                    Column::Bool(v) => {
                        write!(f, "{:<p$}", v[i], p = PADDING)?;
                    }
                }
            }
            writeln!(f)?;
        }

        writeln!(f, "{}", "-".repeat(&self.series.len() * PADDING))?;

        writeln!(f, "\ndtypes:")?;

        for s in &self.series {
            writeln!(f, "{}: {}", s.name, s.column.dtype())?;
        }

        Ok(())
    }
}
