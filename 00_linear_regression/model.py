import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=250):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
    
    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

    def fit(self, X, y, X_test=None, y_test=None):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        self.train_losses = []
        self.val_losses = []

        for i in range(self.iterations):
            y_pred = self.predict(X)
            error = y_pred - y
            loss = self.loss(y, y_pred)

            dw = (1 / len(X)) * np.dot(X.T, error)
            db = (1 / len(X)) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
            self.train_losses.append(loss)

            if X_test is not None and y_test is not None:
                y_test_pred = self.predict(X_test)
                test_loss = self.loss(y_test, y_test_pred)
                self.val_losses.append(test_loss)
                print(f"Iteration {i+1}/{self.iterations}, Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f}")
            else:
                print(f"Iteration {i+1}/{self.iterations}, Train Loss: {loss:.4f}")

        return self
        
    