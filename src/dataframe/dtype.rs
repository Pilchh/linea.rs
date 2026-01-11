use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Dtype {
    Int64,
    Float64,
    String,
    Bool,
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Dtype::Int64 => "Int64",
            Dtype::Float64 => "Float64",
            Dtype::String => "String",
            Dtype::Bool => "Bool",
        };
        write!(f, "{}", s)
    }
}
