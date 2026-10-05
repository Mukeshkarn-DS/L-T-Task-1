import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from nn.layers import Dense, ReLU, Sigmoid, Tanh, Softmax
from nn.model import NeuralNetwork
from nn.optimizers import Adam

# --- 1. LOAD DATA (Subset for speed) ---
base_dir = os.path.dirname(os.path.abspath(__file__))
print("Loading data...")
train_df = pd.read_csv(os.path.join(base_dir, 'DATA', 'mnist_train.csv')).head(20000)
test_df = pd.read_csv(os.path.join(base_dir, 'DATA', 'mnist_test.csv')).head(5000)

X_train = train_df.drop('label', axis=1).values / 255.0
y_train = np.eye(10)[train_df['label'].values]
X_test = test_df.drop('label', axis=1).values / 255.0
y_test = np.eye(10)[test_df['label'].values]

# --- 2. BUILD MODEL ---
def build_model(activation_name):
    model = NeuralNetwork()
    model.add(Dense(784, 128))
    
    if activation_name == 'relu': 
        model.add(ReLU())
    elif activation_name == 'sigmoid': 
        model.add(Sigmoid())
    elif activation_name == 'tanh': 
        model.add(Tanh())
        
    model.add(Dense(128, 10))
    model.add(Softmax())
    return model

# --- 3. TRAINING LOOP ---
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
            
        avg_loss = epoch_loss / n
        train_acc = np.mean(model.predict(X) == np.argmax(y, axis=1))
        
        history['loss'].append(avg_loss)
        history['acc'].append(train_acc)
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {train_acc*100:.2f}%")
        
    return history

# --- 4. RUN EXPERIMENT ---
if __name__ == "__main__":
    activations = ['relu', 'sigmoid', 'tanh']
    histories = {}
    results = {}
    
    for act in activations:
        print(f"\n--- Training with {act.upper()} activation ---")
        model = build_model(act)
        opt = Adam(lr=0.01) # Using the best LR from your grid search
        hist = train(model, opt, X_train, y_train, epochs=10, batch_size=64)
        histories[act] = hist
        
        test_acc = np.mean(model.predict(X_test) == np.argmax(y_test, axis=1))
        results[act] = test_acc
        print(f"--> Test Accuracy ({act}): {test_acc*100:.2f}%")

    # --- 5. PLOT RESULTS ---
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    for name, hist in histories.items():
        plt.plot(hist['loss'], label=name)
    plt.title('Training Loss by Activation Function')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    
    plt.subplot(1, 2, 2)
    for name, hist in histories.items():
        plt.plot(hist['acc'], label=name)
    plt.title('Training Accuracy by Activation Function')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    
    plt.tight_layout()
    plt.savefig('activation_comparison.png')
    plt.show()

    print("\n--- Final Results Summary ---")
    for name, acc in results.items():
        print(f"{name.upper()}: {acc*100:.2f}%")