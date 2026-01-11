use linears::dataframe::DataFrame;

fn main() {
    let mut df = DataFrame::new();

    df.column("a", vec![1, 2, 3])
        .column("b", vec![1.0, 2.0, 3.0])
        .column("c", vec!["1", "2", "3"]);
    println!("{}", df);

    let df_a = df.select(["a", "c"]);
    println!("{}", df_a);

    let mask = &df["b"].eq(2.0);
    println!("{}", mask);
}
