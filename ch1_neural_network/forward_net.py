import numpy as np


class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # initialize weight and bias
        W1 = np.random.randn(I, H) # 2*4
        b1 = np.random.randn(H) # 4
        W2 = np.random.randn(H, O) # 4*3
        b2 = np.random.randn(O) # 3

        # create layers
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # aggregate all weights to list
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


x = np.random.randn(10, 2)
print(x)

model = TwoLayerNet(2, 4, 3)
s = model.predict(x)
print(s)
print(s.shape) # 10*3
