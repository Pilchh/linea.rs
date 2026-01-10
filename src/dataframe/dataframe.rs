use crate::dataframe::{Column, Series};

pub struct DataFrame {
    columns: Vec<Series>,
    nrows: usize,
}
