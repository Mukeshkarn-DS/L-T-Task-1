import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            
    def get_params_and_grads(self):
        params = {}
        grads = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                params[f'W{i}'] = layer.W
                params[f'b{i}'] = layer.b
                grads[f'W{i}'] = layer.dW
                grads[f'b{i}'] = layer.db
        return params, grads
    
    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs, axis=1)