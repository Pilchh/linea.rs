use linears::{
    dataframe::{DataFrame, Dtype, Series},
    math::matrix::Matrix,
    ml::{Dataset, LinearRegression},
};

fn main() {
    let mut df = DataFrame::from_csv(
        "/home/pilchh/dev/rust/linea.rs/test_data/house_prices_small.csv".into(),
    )
    .unwrap();

    let head = df.head(10);
    println!("{}", head);

    // let mut df = DataFrame::new();
    //
    // df.column("a", vec![1, 2, 3])
    //     .column("b", vec![1.0, 2.0, 3.0])
    //     .column("c", vec!["1", "2", "3"]);
    //
    // println!("{}", df);
    //
    // df.remove("b");
    //
    // println!("{}", df);

    // let dataset = Dataset::from_csv("/home/pilchh/dev/rust/linea.rs/test_data/house_prices.csv");
    // println!("{}", dataset);
    //
    // let mut lr = LinearRegression::new();
    // lr.fit(&dataset, 1);
    //
    // let test = Matrix::from_vec(1, 2, vec![1.0, 1700.0]).unwrap();
    // let res = lr.predict(&test);
    // println!("{:#?}", res[0]);
}
