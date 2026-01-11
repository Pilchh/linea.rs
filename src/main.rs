use linears::{
    dataframe::{DataFrame, Dtype},
    math::matrix::Matrix,
    ml::LinearRegression,
};

fn main() {
    let mut df = DataFrame::from_csv("./test_data/house_prices_full.csv".into()).unwrap();

    df = df.select(["sqft_living", "floors", "view", "price"]);
    df.cast_all(Dtype::Int64, Dtype::Float64);

    let x = df.select(["sqft_living", "floors", "view"]);
    let y = df.select(["price"]);

    let mut lr = LinearRegression::new();
    lr.fit(&x, &y);

    // bias, sqft, floors, view
    let predict = Matrix::from_vec(1, 4, vec![1.0, 1200.0, 2.0, 0.0]).unwrap();
    let prediction = lr.predict(&predict);

    println!("{}", prediction[0]);
}
