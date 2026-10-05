import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from nn.layers import Dense, ReLU, Softmax
from nn.model import NeuralNetwork
from nn.optimizers import Adam

# Load Data (Subset for speed)
base_dir = os.path.dirname(os.path.abspath(__file__))
train_df = pd.read_csv(os.path.join(base_dir, 'DATA', 'mnist_train.csv')).head(10000)
test_df = pd.read_csv(os.path.join(base_dir, 'DATA', 'mnist_test.csv')).head(2000)

X_train = train_df.drop('label', axis=1).values / 255.0
y_train = np.eye(10)[train_df['label'].values]
X_test = test_df.drop('label', axis=1).values / 255.0
y_test = np.eye(10)[test_df['label'].values]

def build_model(hidden_neurons=128, activation='relu'):
    model = NeuralNetwork()
    model.add(Dense(784, hidden_neurons))
    if activation == 'relu': model.add(ReLU())
    elif activation == 'sigmoid': 
        # We need to add Sigmoid to layers.py if not there, but let's stick to ReLU for this example
        pass 
    model.add(Dense(hidden_neurons, 10))
    model.add(Softmax())
    return model

def train(model, optimizer, X, y, epochs=3, batch_size=64):
    n = X.shape[0]
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled, y_shuffled = X[indices], y[indices]
        for i in range(0, n, batch_size):
            xb = X_shuffled[i:i+batch_size]
            yb = y_shuffled[i:i+batch_size]
            probs = model.forward(xb)
            grad = (probs - yb) / len(xb)
            model.backward(grad)
            params, grads = model.get_params_and_grads()
            optimizer.step(params, grads)
    test_preds = model.predict(X_test)
    return np.mean(test_preds == np.argmax(y_test, axis=1))

if __name__ == "__main__":
    # Grid of Hyperparameters
    learning_rates = [0.1, 0.01, 0.001]
    batch_sizes = [16, 64, 128]
    hidden_neurons = [32, 128, 256]
    
    results = []
    
    print("Starting Hyperparameter Tuning...")
    for lr in learning_rates:
        for bs in batch_sizes:
            for hn in hidden_neurons:
                print(f"Testing lr={lr}, bs={bs}, neurons={hn}...", end=" ")
                model = build_model(hidden_neurons=hn, activation='relu')
                opt = Adam(lr=lr)
                acc = train(model, opt, X_train, y_train, epochs=3, batch_size=bs)
                results.append({'lr': lr, 'bs': bs, 'neurons': hn, 'acc': acc})
                print(f"Acc: {acc*100:.2f}%")

    # Save results to a CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('hyperparameter_results.csv', index=False)
    print("\nSaved results to hyperparameter_results.csv")
    print(results_df.sort_values(by='acc', ascending=False).head())