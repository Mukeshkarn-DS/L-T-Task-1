import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from nn.layers import Dense, ReLU, Softmax
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

# --- 2. BUILD AND TRAIN MODEL ---
def build_model():
    model = NeuralNetwork()
    model.add(Dense(784, 128))
    model.add(ReLU())
    model.add(Dense(128, 10))
    model.add(Softmax())
    return model

def train(model, optimizer, X, y, epochs=10, batch_size=64):
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
    return model

if __name__ == "__main__":
    print("Training model (this may take a minute)...")
    model = build_model()
    opt = Adam(lr=0.01) # Best LR from your grid search
    model = train(model, opt, X_train, y_train, epochs=10, batch_size=64)
    
    # --- 3. MAKE PREDICTIONS ---
    test_preds = model.predict(X_test)
    true_labels = np.argmax(y_test, axis=1)
    
    # --- 4. CONFUSION MATRIX (Pure NumPy) ---
    print("\nGenerating Confusion Matrix...")
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(true_labels, test_preds):
        cm[t, p] += 1
        
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix on Test Data')
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add numbers inside the cells
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', 
                     color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # --- 5. MISCLASSIFIED IMAGES ---
    misclassified_idx = np.where(test_preds != true_labels)[0]
    print(f"Total misclassified: {len(misclassified_idx)} out of {len(true_labels)}")
    
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(misclassified_idx[:10]): # Show first 10 errors
        plt.subplot(2, 5, i+1)
        img = X_test[idx].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {true_labels[idx]}, Pred: {test_preds[idx]}")
        plt.axis('off')
    plt.suptitle('Examples of Misclassified Images', fontsize=16)
    plt.tight_layout()
    plt.savefig('misclassified_images.png')
    plt.show()
    
    print("\nSaved confusion_matrix.png and misclassified_images.png")