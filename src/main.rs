use linears::dataframe::DataFrame;

fn main() {
    let mut df = DataFrame::from_csv("./test_data/house_prices_full.csv".into()).unwrap();

    println!(
        "{}",
        df.head(10)
            .select(["sqft_living", "floors", "view", "price"])
    );

    df.cast_all(
        linears::dataframe::Dtype::Int64,
        linears::dataframe::Dtype::Float64,
    );

    println!(
        "{}",
        df.head(10)
            .select(["sqft_living", "floors", "view", "price"])
    );
}
