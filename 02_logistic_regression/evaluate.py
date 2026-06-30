from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from utils import load_model, load_data
import matplotlib.pyplot as plt

model = load_model("saved_models/logistic_regression.pkl")
scaler = load_model("saved_models/scaler.pkl")

_, X_test, _, y_test = load_data()
X_test = scaler.transform(X_test)

y_preds = model.predict_class(X_test)

accuracy = accuracy_score(y_test, y_preds)
precision = precision_score(y_test, y_preds)
recall = recall_score(y_test, y_preds)
f1 = f1_score(y_test, y_preds)
cm = confusion_matrix(y_test, y_preds)

print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1 Score : {f1:.3f}")

plt.figure("Training Loss")
plt.plot(range(1, len(model.losses) + 1), model.losses)
plt.title('Logistic Regression Training Loss Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid()
plt.tight_layout()
plt.savefig("results/loss_curve.png", dpi=300)

plt.figure("Confusion Matrix")
fig, ax = plt.subplots(num="Confusion Matrix", clear=True)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300)

plt.show()
