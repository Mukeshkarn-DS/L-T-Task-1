import numpy as np

class Dense:
    """Fully connected (Linear) layer."""
    def __init__(self, in_dim, out_dim):
        # He initialization for ReLU networks
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros((1, out_dim))
        
        # Cache for backward pass
        self.dW = None
        self.db = None
        self.input = None

    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.W) + self.b
        return self.z

    def backward(self, grad):
        # grad is dL/dz
        self.dW = np.dot(self.input.T, grad)
        self.db = np.sum(grad, axis=0, keepdims=True)
        return np.dot(grad, self.W.T)


class ReLU:
    """Rectified Linear Unit activation function."""
    def __init__(self):
        self.mask = None

    def forward(self, z):
        self.mask = z > 0
        return z * self.mask

    def backward(self, grad):
        return grad * self.mask


class Sigmoid:
    """Sigmoid activation function."""
    def __init__(self):
        self.out = None

    def forward(self, z):
        # Clip z to prevent overflow in np.exp
        z_clipped = np.clip(z, -500, 500)
        self.out = 1 / (1 + np.exp(-z_clipped))
        return self.out

    def backward(self, grad):
        return grad * (self.out * (1 - self.out))


class Tanh:
    """Hyperbolic Tangent activation function."""
    def __init__(self):
        self.out = None

    def forward(self, z):
        self.out = np.tanh(z)
        return self.out

    def backward(self, grad):
        return grad * (1 - self.out**2)


class Softmax:
    """Softmax activation function (usually used in the output layer)."""
    def __init__(self):
        self.probs = None

    def forward(self, z):
        # Subtract max for numerical stability (prevents overflow)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.probs

    def backward(self, grad):
        # Note: In this implementation, we combine Softmax + Cross Entropy Loss.
        # The gradient of the combined loss is simply (probs - y_true) / batch_size.
        # So we just pass the gradient through this layer.
        return grad