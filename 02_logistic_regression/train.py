from utils import save_model, load_data, scale_data
from model import LogisticRegression

X_train, X_test, y_train, y_test = load_data()
X_train, X_test, scaler = scale_data(X_train, X_test)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

logistic_regression = LogisticRegression(learning_rate=0.01, iterations=300)
logistic_regression.fit(X_train, y_train)

save_model(logistic_regression, "saved_models/logistic_regression.pkl")
save_model(scaler, "saved_models/scaler.pkl")