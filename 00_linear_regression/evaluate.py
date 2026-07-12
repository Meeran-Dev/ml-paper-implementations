from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from utils import load_model, load_data
import matplotlib.pyplot as plt

model = load_model("saved_models/linear_regression.pkl")
scaler = load_model("saved_models/scaler.pkl")

X_train, X_test, y_train, y_test = load_data()
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

y_preds = model.predict(X_test)

mse = mean_squared_error(y_test, y_preds)
r2 = r2_score(y_test, y_preds)

print("Implemented Model Evaluation:")
print(f"Mean Squared Error: {mse:.3f}")
print(f"R^2 Score: {r2:.3f}")

sk_model = LinearRegression()
sk_model.fit(X_train, y_train)
y_sk_preds = sk_model.predict(X_test)
sk_mse = mean_squared_error(y_test, y_sk_preds)
sk_r2 = r2_score(y_test, y_sk_preds)

print("\nScikit-learn Model Evaluation:")
print(f"Mean Squared Error: {sk_mse:.3f}")
print(f"R^2 Score: {sk_r2:.3f}")

plt.figure("Loss Curve")
plt.plot(range(1, len(model.train_losses) + 1), model.train_losses)
plt.plot(range(1, len(model.val_losses) + 1), model.val_losses)
plt.legend(['Train Loss', 'Validation Loss'])
plt.title('Linear Regression Loss Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid()
plt.tight_layout()
plt.savefig("results/loss_curve.png", dpi=300)

plt.show()