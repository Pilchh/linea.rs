use crate::{
    dataframe::{DataFrame, Series},
    math::{matrix::Matrix, vector::Vector},
    ml::dataset::Dataset,
};

pub struct LinearRegression {
    pub coefficients: Option<Vector>,
}

impl LinearRegression {
    pub fn new() -> Self {
        LinearRegression { coefficients: None }
    }

    pub fn fit(&mut self, x: &DataFrame, y: &DataFrame) {
        let (rows, _) = x.shape();

        let mut x_df = x.clone();

        // Insert bias column
        let bias = Series::from_vec("bias".into(), vec![1.0; rows]);
        x_df.insert(0, bias);

        let mut x_mat = x_df.as_matrix();

        // Get the label columns
        let y = y.clone().as_vector();

        // Calculate coefficients
        let (q, r) = x_mat.qr_decompose().unwrap();
        let qt_y = &q.transpose() * &y;

        self.coefficients = Some(r.solve_upper(&qt_y).unwrap());
    }

    pub fn predict(&self, x: &Matrix) -> Vector {
        x * self.coefficients.as_ref().expect("model not fitted")
    }
}
