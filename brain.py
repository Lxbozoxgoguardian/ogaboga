import numpy as np

class NeuralNet:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        assert input_size > 0 and hidden_size > 0 and output_size > 0

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, x):
        assert x.shape[1] == self.input_size

        z1 = x @ self.w1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.w2 + self.b2
        out = self.softmax(z2)

        return out, a1
