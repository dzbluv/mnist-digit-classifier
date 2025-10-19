import pickle
import numpy as np
import time

# ---------- Helpers ----------
def sig(x):
    return 1 / (1 + np.exp(-x))

def predict(x, weights, bias):
    return sig(np.dot(x, weights) + bias)

def cost(y_true, y_pred):
    eps = 1e-10
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# ---------- Mini-batch Gradient Descent ----------
def gra_dec_minibatch(x_train, y_train, lr, epochs, batch_size):
    n_samples, n_features = x_train.shape
    weights = np.zeros(n_features)
    bias = 0
    initial_lr = lr

    for epoch in range(epochs):
        # Learning rate decay
        lr = initial_lr / (1 + 0.001 * epoch)
        
        # Shuffle data each epoch
        indices = np.random.permutation(n_samples)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]

        # Mini-batch updates
        for i in range(0, n_samples, batch_size):
            x_batch = x_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]

            y_pred = sig(np.dot(x_batch, weights) + bias)

            # Gradients
            dw = (1 / len(y_batch)) * np.dot(x_batch.T, (y_pred - y_batch))
            db = (1 / len(y_batch)) * np.sum(y_pred - y_batch)

            # Parameter update
            weights -= lr * dw
            bias -= lr * db

        # Progress log
        if (epoch + 1) % 50 == 0:
            c = cost(y_train, predict(x_train, weights, bias))
            print(f"[INFO] Epoch {epoch+1:04d} - cost: {c:.5f}")

    return weights, bias

# ---------- Data Loading ----------
def loaddata(fname):
    labels, data = [], []
    with open(fname, 'r') as f:
        next(f)
        for line in f:
            values = line.strip().split(',')
            labels.append(int(values[0]))
            pixels = [int(v)/255 for v in values[1:]]
            data.append(pixels)
    return np.array(data), np.array(labels)

def bin_labels(labels, val):
    return np.array([1 if l == val else 0 for l in labels])

# ---------- Main ----------
if __name__ == "__main__":
    print("[INFO] Loading training data...")
    x_train, y_train = loaddata('mnist_train.csv')
    x_train = np.array(x_train)
    print("[INFO] Data loaded successfully.")
    
    start_time = time.time()
    models = []

    for val in range(10):
        print(f"\n[INFO] Training model for digit {val}...")
        y_train_bin = bin_labels(y_train, val)

        lr = 0.05
        epochs = 500
        batch_size = 64

        weights, bias = gra_dec_minibatch(x_train, y_train_bin, lr, epochs, batch_size)
        models.append({"weights": weights, "bias": bias})
        print(f"[INFO] Finished training digit {val}.")
        print(f"       Final bias: {bias:.6f}")
        print(f"       Sample weights [0:10]: {weights[:10]}")
        print(f"       Sample weights [350:360]: {weights[350:360]}")

    # Save all models
    with open("mnist_model_cpu_debug.pkl", "wb") as f:
        pickle.dump(models, f)

    end_time = time.time()
    print(f"\n[INFO] All 10 models saved to mnist_model_cpu_debug.pkl")
    print(f"[INFO] Total training time: {end_time - start_time:.2f}s")
