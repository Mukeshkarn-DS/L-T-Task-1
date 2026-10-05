import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from nn.layers import Dense, ReLU, Softmax
from nn.model import NeuralNetwork
from nn.optimizers import Adam

# Load Data (Small subset for speed)
base_dir = os.path.dirname(os.path.abspath(__file__))
train_df = pd.read_csv(os.path.join(base_dir, 'DATA', 'mnist_train.csv')).head(10000)
test_df = pd.read_csv(os.path.join(base_dir, 'DATA', 'mnist_test.csv')).head(2000)

X_train = train_df.drop('label', axis=1).values / 255.0
y_train = np.eye(10)[train_df['label'].values]
X_test = test_df.drop('label', axis=1).values / 255.0
y_test = np.eye(10)[test_df['label'].values]

def train(model, optimizer, X, y, epochs=10, batch_size=64):
    n = X.shape[0]
    history = {'loss': [], 'acc': []}
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled, y_shuffled = X[indices], y[indices]
        epoch_loss = 0
        for i in range(0, n, batch_size):
            xb = X_shuffled[i:i+batch_size]
            yb = y_shuffled[i:i+batch_size]
            probs = model.forward(xb)
            loss = -np.mean(np.sum(yb * np.log(probs + 1e-8), axis=1))
            epoch_loss += loss * len(xb)
            grad = (probs - yb) / len(xb)
            model.backward(grad)
            params, grads = model.get_params_and_grads()
            optimizer.step(params, grads)
        history['loss'].append(epoch_loss / n)
        history['acc'].append(np.mean(model.predict(X) == np.argmax(y, axis=1)))
    return history

if __name__ == "__main__":
    histories = {}
    
    # 1. Single Layer Model (No hidden layer)
    print("Training Single Layer Network...")
    single_model = NeuralNetwork()
    single_model.add(Dense(784, 10)) # Directly to output
    single_model.add(Softmax())
    opt_single = Adam(lr=0.01)
    histories['Single Layer'] = train(single_model, opt_single, X_train, y_train, epochs=10, batch_size=64)
    single_acc = np.mean(single_model.predict(X_test) == np.argmax(y_test, axis=1))
    
    # 2. Multi Layer Model (1 hidden layer)
    print("Training Multi Layer Network...")
    multi_model = NeuralNetwork()
    multi_model.add(Dense(784, 128)) # Hidden layer
    multi_model.add(ReLU())
    multi_model.add(Dense(128, 10))
    multi_model.add(Softmax())
    opt_multi = Adam(lr=0.01)
    histories['Multi Layer'] = train(multi_model, opt_multi, X_train, y_train, epochs=10, batch_size=64)
    multi_acc = np.mean(multi_model.predict(X_test) == np.argmax(y_test, axis=1))

    # Plot
    plt.figure(figsize=(10, 4))
    for name, hist in histories.items():
        plt.plot(hist['acc'], label=name, marker='o')
    plt.title('Single Layer vs Multi Layer Network')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.savefig('single_vs_multi.png')
    plt.show()

    print(f"\nSingle Layer Test Accuracy: {single_acc*100:.2f}%")
    print(f"Multi Layer Test Accuracy:  {multi_acc*100:.2f}%")