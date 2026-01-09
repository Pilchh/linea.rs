# Linea.rs
### Project Structure

## Data
- Dataset struct: holds features and labels.
- Preprocessing tools: normalization, train/test split, feature scaling.

## Math
**Matrix**
- `new_filled` - Creates a new matrix filled with a defined value.
- `from_vec` - Creates a new matrix from a vector.
- `get` - Gets the value from a matrix at an index.
- `set` - Sets a matrix value at index.
- `row` - Gets an entire row from the matrix.
- `col` - Gets an entire column from the matrix.
- `is_square` - Checks if a matrix is square.
- `add` - Adds two matrices.
- `multiply` - Multiplies to matrices.
- `transpose` - Transposes a matrix

**Vector**
- TODO

## Metrics
- Regression metrics: MSE, MAE, R².

## Model
- Linear regression implementation: core LinearRegression struct, fit/predict methods.
- Regularisation support: optional L1/L2 logic.

## Optim
- Gradient descent: iterative training logic.
- Normal equation solver: closed-form solution for small datasets.
