import cupy as cp
import pickle
import time

cp.random.seed(0)

# ---------- Helpers ----------
def softmax(z):
    z = z - cp.max(z, axis=1, keepdims=True)
    exp_z = cp.exp(z)
    return exp_z / cp.sum(exp_z, axis=1, keepdims=True)

def one_hot(y, num_classes=10):
    # Convert integer labels to one-hot encoded format
    return cp.eye(num_classes, dtype=cp.float64)[y]

def loaddata(fname):
    labels, data = [], []
    with open(fname, 'r') as f:
        next(f)
        for line in f:
            vals = line.strip().split(',')
            labels.append(int(vals[0]))
            data.append([int(v)/255 for v in vals[1:]])
    X = cp.array(data, dtype=cp.float64)
    y = cp.array(labels, dtype=cp.int64)
    return X, y

def cross_entropy_loss(y_true_onehot, y_pred_probs, W=None, l2=0.0):
    eps = 1e-12
    logp = cp.log(y_pred_probs + eps)
    data_loss = -cp.mean(cp.sum(y_true_onehot * logp, axis=1))
    if l2 and (W is not None):
        reg = (l2 / (2.0 * y_true_onehot.shape[0])) * cp.sum(W * W)
        return data_loss + reg
    return data_loss

def accuracy_from_probs(y_true_int, probs):
    preds = cp.argmax(probs, axis=1)
    return float(cp.mean(preds == y_true_int))

# ---------- Training (Multiclass Logistic Regression) ----------
def train_logreg_fullbatch(X, y_int, lr, epochs, l2, use_momentum, beta, decay):
    N, D = X.shape
    C = 10

    # Initialize small random weights
    W = cp.random.randn(D, C).astype(cp.float64) * 0.01
    b = cp.zeros((1, C), dtype=cp.float64)

    y_onehot = one_hot(y_int, C)

    # Momentum buffers
    vW = cp.zeros_like(W)
    vb = cp.zeros_like(b)

    initial_lr = lr

    # Initial metrics
    logits = X @ W + b
    probs = softmax(logits)
    init_loss = cross_entropy_loss(y_onehot, probs, W, l2)
    init_acc = accuracy_from_probs(y_int, probs)
    print(f"[INFO] Initial loss: {float(init_loss):.6f}, accuracy: {init_acc*100:.2f}%")

    for epoch in range(1, epochs + 1):
        # Forward pass
        logits = X @ W + b
        probs = softmax(logits)

        # Compute gradients
        dlogits = (probs - y_onehot) / N
        dW = X.T @ dlogits
        if l2:
            dW += (l2 / N) * W
        db = cp.sum(dlogits, axis=0, keepdims=True)

        # Apply momentum if enabled
        if use_momentum:
            vW = beta * vW + (1 - beta) * dW
            vb = beta * vb + (1 - beta) * db
            W -= lr * vW
            b -= lr * vb
        else:
            W -= lr * dW
            b -= lr * db

        # Learning rate decay
        lr = initial_lr / (1.0 + decay * epoch)

        if epoch % 50 == 0 or epoch == 1:
            loss = cross_entropy_loss(y_onehot, probs, W, l2)
            acc = accuracy_from_probs(y_int, probs)
            print(f"[INFO] Epoch {epoch:04d} - loss: {float(loss):.6f}, acc: {acc*100:.2f}%")

    return W, b

# ---------- Main ----------
if __name__ == "__main__":
    print("[INFO] Loading training data...")
    X, y = loaddata('mnist_train.csv')

    # Standardize features
    X = (X - cp.mean(X, axis=0)) / (cp.std(X, axis=0) + 1e-8)
    print(f"[INFO] X shape: {X.shape}, dtype: {X.dtype}")
    print(f"[INFO] y shape: {y.shape}, dtype: {y.dtype}")

    start = time.time()
    W, b = train_logreg_fullbatch(
        X, y,
        lr=0.1,
        epochs=1000,
        l2=0.0001,
        use_momentum=True,
        beta=0.9,
        decay=0.0005
    )
    end = time.time()
    print(f"[INFO] Training completed in {end - start:.2f}s")

    # Save model
    model = {"W": cp.asnumpy(W), "b": cp.asnumpy(b)}
    with open("mnist_logreg_gpu_debug.pkl", "wb") as f:
        pickle.dump(model, f)
    print("[INFO] Model saved to mnist_logreg_gpu_debug.pkl")
