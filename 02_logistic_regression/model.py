import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=100):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def weighted_sum(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def predict(self, inputs):
        weighted_sum = self.weighted_sum(inputs)
        return self.sigmoid(weighted_sum)

    def predict_class(self, inputs, threshold=0.5):
        probabilities = self.predict(inputs)
        return np.where(probabilities >= threshold, 1, 0)
    
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        self.losses = []

        for i in range(self.iterations):

            y_pred = self.predict(X)
            error = y_pred - y
            loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))

            dw = (1/len(X)) * X.T @ error
            db = (1/len(X)) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            print(f"Iteration {i+1}/{self.iterations}, Loss: {loss:.4f}")
            self.losses.append(loss)
        
        return self