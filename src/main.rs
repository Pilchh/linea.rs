use linears::{
    dataframe::DataFrame,
    math::matrix::Matrix,
    ml::{Dataset, LinearRegression},
};

fn main() {
    // let df = DataFrame::from_csv(
    //     "/home/pilchh/dev/rust/linea.rs/test_data/house_prices_small.csv".into(),
    // )
    // .unwrap();
    //
    // let head = df.head(10);
    // println!("{}", head);

    let mut df = DataFrame::new();

    df.column("a", vec![1, 2, 3])
        .column("b", vec![1.0, 2.0, 3.0])
        .column("c", vec!["1", "2", "3"]);
    println!("{}", df);

    let df_a = df.select(["a", "c"]);
    println!("{}", df_a);

    let series = &df["b"].eq(2.0);
    println!("{}", series);
    // let dataset =
    //     Dataset::from_csv("/home/pilchh/dev/rust/linea.rs/test_data/house_prices_small.csv");
    // println!("{}", dataset);
    //
    // let mut lr = LinearRegression::new();
    // lr.fit(&dataset, 1);
    //
    // let test = Matrix::from_vec(1, 2, vec![1.0, 700.0]).unwrap();
    // let res = lr.predict(&test);
    // println!("{:#?}", res[0]);
}
