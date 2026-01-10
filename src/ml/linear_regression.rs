use crate::{math::matrix::Matrix, math::vector::Vector, ml::dataset::Dataset};

pub struct LinearRegression {
    pub coefficients: Option<Vector>,
}

impl LinearRegression {
    pub fn new() -> Self {
        LinearRegression { coefficients: None }
    }

    pub fn fit(&mut self, dataset: &Dataset, target_col: usize) {
        // Get all data
        let mut x = dataset.samples();

        // Get the label columns
        let y = Vector::from_vec(x.col(target_col).unwrap());

        // Remove the label column
        x = x.drop_col(target_col);

        // Calculate coefficients
        let (q, r) = x.qr_decompose().unwrap();
        let qt_y = &q.transpose() * &y;

        self.coefficients = Some(r.solve_upper(&qt_y).unwrap());
    }

    pub fn predict(&self, x: &Matrix) -> Vector {
        x * self.coefficients.as_ref().expect("model not fitted")
    }
}
