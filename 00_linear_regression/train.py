from utils import save_model, load_data, scale_data
from model import LinearRegression

X_train, X_test, y_train, y_test = load_data()
X_train, X_test, scaler = scale_data(X_train, X_test)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

linear_regression = LinearRegression(learning_rate=0.01, iterations=250)
linear_regression.fit(X_train, y_train, X_test, y_test)

save_model(linear_regression, "saved_models/linear_regression.pkl")
save_model(scaler, "saved_models/scaler.pkl")