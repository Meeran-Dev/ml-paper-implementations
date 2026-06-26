from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import load_model, load_data
import matplotlib.pyplot as plt

model = load_model("saved_models/perceptron.pkl")
_, X_test, _, y_test = load_data()

y_preds = model.predict(X_test)

accuracy = accuracy_score(y_test, y_preds)
precision = precision_score(y_test, y_preds)
recall = recall_score(y_test, y_preds)
f1 = f1_score(y_test, y_preds)

print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1 Score : {f1:.3f}")

plt.plot(range(1, len(model.errors) + 1), model.errors)
plt.title('Perceptron Training Errors Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Errors') 
plt.grid()
plt.tight_layout()
plt.savefig("results/error_curve.png", dpi=300)
plt.show()