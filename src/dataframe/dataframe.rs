use std::{fmt, ops::Index};

use crate::dataframe::{Column, Dtype, Series, column::IntoColumn, errors::DataFrameError};

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

    pub fn from_series(series: Vec<Series>) -> Result<DataFrame, DataFrameError> {
        if series.len() == 0 {
            return Err(DataFrameError::InvalidDataProvided);
        }

        let are_same_length = series
            .first()
            .map(|first| series.iter().all(|s| s.len() == first.len()))
            .unwrap_or(true);

        if !are_same_length {
            return Err(DataFrameError::SeriesLengthMismatch);
        }

        let nrows = series[0].len();
        Ok(DataFrame { series, nrows })
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
                return Err(DataFrameError::SeriesDataFrameLengthMismatch {
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
        if let Some(position) = self.series.iter().position(|s| s.name == name) {
            self.remove_series(position);
        }

        self
    }

    pub fn select<'a, I>(&self, names: I) -> DataFrame
    where
        I: IntoIterator<Item = &'a str>,
    {
        let series: Vec<Series> = names
            .into_iter()
            .map(|name| self.series(name).clone())
            .collect();

        DataFrame::from_series(series).unwrap()
    }

    pub fn filter(&self, mask: &Series) -> DataFrame {
        if mask.dtype != Dtype::Bool {
            panic!("filter can only use boolean series");
        }

        if mask.column.len() != self.series[0].len() {
            panic!("mask length does not match dataframe length");
        }

        let filtered_series: Vec<Series> = self.series.iter().map(|s| s.filter(mask)).collect();

        DataFrame::from_series(filtered_series).unwrap()
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
        (self.nrows, self.series.len())
    }

    fn assert_compatible_series(&mut self, series: &Series) {
        let series_len = series.len();

        // If this is the first series being added
        if self.series.is_empty() {
            if series_len == 0 {
                panic!("Cannot add empty series as the first column");
            }
            self.nrows = series_len;
            return;
        }

        // For all subsequent series, lengths must match
        if series_len != self.nrows {
            panic!(
                "Series length mismatch: DataFrame has {} rows, series has {}",
                self.nrows, series_len
            );
        }
    }

    fn series(&self, name: &str) -> &Series {
        self.series
            .iter()
            .find(|s| s.name == name)
            .expect("Series does not exist in dataframe")
    }

    fn get_series(&self, name: &str) -> Series {
        self.series(name).clone()
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

impl Index<&str> for DataFrame {
    type Output = Series;

    fn index(&self, name: &str) -> &Self::Output {
        self.series
            .iter()
            .find(|s| s.name == name)
            .expect("series not found")
    }
}

const PADDING: usize = 15;

impl fmt::Display for DataFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.series.len() == 0 {
            writeln!(f, "Empty Dataframe")?;
            return Ok(());
        }

        let shape = self.shape();
        writeln!(f, "Height: {}, Width: {}", shape.0, shape.1)?;
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

        writeln!(f, "dtypes:")?;

        for s in &self.series {
            writeln!(f, "{}: {}", s.name, s.column.dtype())?;
        }

        writeln!(f, "{}", "-".repeat(&self.series.len() * PADDING))?;

        Ok(())
    }
}
