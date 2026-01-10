use linears::{
    matrix::Matrix,
    ml::{dataset::Dataset, linear_regression::LinearRegression},
};

fn main() {
    let mut dataset =
        Dataset::from_csv("/home/pilchh/dev/rust/linea.rs/test_data/house_prices.csv");

    let mut lr = LinearRegression::new();
    lr.fit(&dataset, 1);

    let test = Matrix::from_vec(1, 2, vec![1.0, 700.0]).unwrap();
    let res = lr.predict(&test);
    println!("{:#?}", res[0]);
}
