pub enum Scalar {
    Int64(i64),
    Float64(f64),
    String(String),
    Bool(bool),
}

impl From<i64> for Scalar {
    fn from(v: i64) -> Self {
        Scalar::Int64(v)
    }
}

impl From<f64> for Scalar {
    fn from(v: f64) -> Self {
        Scalar::Float64(v)
    }
}

impl From<&str> for Scalar {
    fn from(v: &str) -> Self {
        Scalar::String(v.into())
    }
}

impl From<bool> for Scalar {
    fn from(v: bool) -> Self {
        Scalar::Bool(v)
    }
}
