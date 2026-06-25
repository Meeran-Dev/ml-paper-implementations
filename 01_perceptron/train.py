from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import model

breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

perceptron = model.Perceptron(learning_rate=0.01, iterations=200)
perceptron.fit(X_train, y_train)

y_preds = perceptron.predict(X_test)

accuracy = accuracy_score(y_preds, y_test)
print('Accuracy: {:.2f}%'.format(accuracy * 100))