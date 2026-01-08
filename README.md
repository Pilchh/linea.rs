# Linea.rs
### Project Structure

Data
- Dataset struct: holds features and labels.
- Preprocessing tools: normalization, train/test split, feature scaling.

Math
- Matrix operations: multiplication, transpose, dot products.
- Vector utilities: basic vector math.

Metrics
- Regression metrics: MSE, MAE, R².

Model
- Linear regression implementation: core LinearRegression struct, fit/predict methods.
- Regularisation support: optional L1/L2 logic.

Optim
- Gradient descent: iterative training logic.
- Normal equation solver: closed-form solution for small datasets.
