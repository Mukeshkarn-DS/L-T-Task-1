import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Import our custom modules
from nn.layers import Dense, ReLU, Softmax
from nn.model import NeuralNetwork
from nn.optimizers import SGD, Momentum, RMSProp, Adam

# --- 1. LOAD DATA ---
base_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_dir, 'DATA', 'mnist_train.csv')
test_path = os.path.join(base_dir, 'DATA', 'mnist_test.csv')

print("Loading data...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop('label', axis=1).values / 255.0
y_train = np.eye(10)[train_df['label'].values]

X_test = test_df.drop('label', axis=1).values / 255.0
y_test = np.eye(10)[test_df['label'].values]

print(f"Data loaded. Training shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- 2. BUILD MODEL ---
def build_model(hidden_neurons=128):
    model = NeuralNetwork()
    model.add(Dense(784, hidden_neurons))
    model.add(ReLU())
    model.add(Dense(hidden_neurons, 10))
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
            
            # Forward pass
            probs = model.forward(xb)
            
            # Cross-entropy loss
            loss = -np.mean(np.sum(yb * np.log(probs + 1e-8), axis=1))
            epoch_loss += loss * len(xb)
            
            # Backward pass (Gradient of Cross-Entropy + Softmax)
            grad = (probs - yb) / len(xb)
            model.backward(grad)
            
            # Update weights
            params, grads = model.get_params_and_grads()
            optimizer.step(params, grads)
            
        # Calculate epoch metrics
        avg_loss = epoch_loss / n
        train_preds = model.predict(X)
        train_acc = np.mean(train_preds == np.argmax(y, axis=1))
        
        history['loss'].append(avg_loss)
        history['acc'].append(train_acc)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        
    return history

# --- 4. RUN EXPERIMENT ---
if __name__ == "__main__":
    # Build Model
    model = build_model(hidden_neurons=128)
    
    # Choose Optimizer (Change this to try SGD, Momentum, RMSProp, or Adam)
    # Recommended starting point: Adam with lr=0.001
    optimizer = Adam(lr=0.001)
    
    # Train
    print("\nStarting training...")
    history = train(model, optimizer, X_train, y_train, epochs=10, batch_size=64)
    
    # Evaluate on Test Data
    test_preds = model.predict(X_test)
    test_acc = np.mean(test_preds == np.argmax(y_test, axis=1))
    print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
    
    # --- 5. VISUALIZE ---
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['acc'])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Saved training_curves.png")
    plt.show()