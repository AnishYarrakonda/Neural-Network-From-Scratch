import numpy as np
import matplotlib.pyplot as plt
from dense import Dense
from activations import Tanh, Softmax
from loss import CCE
from model import NNModel
from helper import plot_decision_boundary, plot_accuracy_curve


# Spiral dataset generator
def spiral_data(points: int, classes: int, radius: float, noise: float) -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, radius, points)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * noise
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


# Prepare data
X, y = spiral_data(100, 3, 1.0, 0.1)

# shuffle data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]

# split 80/20
split = int(0.8 * X.shape[0])
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# one-hot encode for CCE
y_train_onehot = np.zeros((y_train.size, y_train.max() + 1))
y_train_onehot[np.arange(y_train.size), y_train] = 1

y_val_onehot = np.zeros((y_val.size, y_val.max() + 1))
y_val_onehot[np.arange(y_val.size), y_val] = 1

# Build the model
model = NNModel()
model.add(Dense(2, 128))  # input -> 128
model.add(Tanh())
model.add(Dense(128, 128))  # hidden -> 128
model.add(Tanh())
model.add(Dense(128, 64))  # hidden -> 64
model.add(Tanh())
model.add(Dense(64, 3))  # output -> 3 classes
model.add(Softmax())
model.set_loss(CCE())

# Training loop with animation
epochs = 800                            # enough for spiral convergence
lr = 0.05                               # higher learning rate
train_losses, val_losses = [], []
train_acc_history, val_acc_history = [], []

plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))

for epoch in range(epochs):
    # --- forward & backward ---
    y_pred = model.forward(X_train)
    loss = model.loss.forward(y_pred, y_train_onehot)
    train_losses.append(loss)
    model.backward(y_pred, y_train_onehot, lr)

    # --- validation loss ---
    y_val_pred = model.forward(X_val)
    val_loss = model.loss.forward(y_val_pred, y_val_onehot)
    val_losses.append(val_loss)

    # --- compute accuracies ---
    y_pred_class = np.argmax(y_pred, axis=1)
    y_val_class = np.argmax(y_val_pred, axis=1)
    train_acc_history.append(np.mean(y_pred_class == y_train))
    val_acc_history.append(np.mean(y_val_class == y_val))

    # --- update decision boundary animation every 5 epochs ---
    if epoch % 5 == 0 or epoch == epochs - 1:
        ax.clear()
        plot_decision_boundary(model, X_train, y_train, X_val, y_val, ax=ax,
                               title=f'Epoch {epoch + 1}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')
        plt.pause(0.01)

plt.ioff()
plt.show()

# Plot training & validation accuracy
plot_accuracy_curve(train_acc_history, val_acc_history)