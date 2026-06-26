from utils import save_model, load_data
from model import Perceptron

X_train, X_test, y_train, y_test = load_data()

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

perceptron = Perceptron(learning_rate=0.01, iterations=250)
perceptron.fit(X_train, y_train)

save_model(perceptron, "saved_models/perceptron.pkl")