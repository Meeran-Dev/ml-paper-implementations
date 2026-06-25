import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, iterations=100):
        self.iterations = iterations
        self.learning_rate = learning_rate
    
    def weighted_sum(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def predict(self, inputs):
        weighted_sum = self.weighted_sum(inputs)
        return np.where(weighted_sum >= 0, 1, 0)

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        self.errors = []

        for _ in range(self.iterations):
            error = 0

            for Xi, yi in zip(X, y):
                y_pred = self.predict(Xi)
                update = self.learning_rate * (yi - y_pred)
                self.weights += update * Xi
                self.bias += update
                error += int(update != 0.0)
                
            print(f"Iteration {_+1}/{self.iterations}, Error: {error}")
            self.errors.append(error)

        return self