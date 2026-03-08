# imports
from dense import Dense
from activations import *
from loss import *
from model import NNModel
from helper import *


# ── Hyperparameters ────────────────────────────────────────────────────────────

# dataset
POINTS      = 300           # number of points per class
CLASSES     = 3             # number of classes
RADIUS      = 1.0           # maximum radius of the spiral
NOISE       = 0.1           # noise added to the spiral angles
VAL_SPLIT   = 0.8           # fraction of data used for training (rest is validation)

# architecture — add/remove entries to change depth, edit values to change width
HIDDEN_LAYERS   = [256, 256, 128, 64]   # neurons in each hidden layer
DROPOUT_RATES   = [0.3, 0.25, 0.2]      # dropout rate after each hidden layer (must be len(HIDDEN_LAYERS) - 1 or less)

# training
EPOCHS      = 1000          # total number of training epochs
LR          = 0.02          # base learning rate (momentum handles acceleration)
BATCH_SIZE  = 32            # mini-batch size
MOMENTUM    = 0.9           # momentum coefficient for weight updates
PRINT_EVERY = 50            # print accuracy to console every N epochs

# ──────────────────────────────────────────────────────────────────────────────

# Prepare data
X, y = spiral_data(POINTS, CLASSES, RADIUS, NOISE)

# shuffle data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]

# split 80/20
split = int(VAL_SPLIT * X.shape[0])
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# one-hot encode for CCE
y_train_onehot = np.zeros((y_train.size, y_train.max() + 1))
y_train_onehot[np.arange(y_train.size), y_train] = 1

y_val_onehot = np.zeros((y_val.size, y_val.max() + 1))
y_val_onehot[np.arange(y_val.size), y_val] = 1

# Build the model — loops through HIDDEN_LAYERS and DROPOUT_RATES automatically
model = NNModel()
layer_sizes = [2] + HIDDEN_LAYERS                       # insert input size at the front so we can zip adjacent pairs
for i, (n_in, n_out) in enumerate(zip(layer_sizes, layer_sizes[1:])):
    model.add(Dense(n_in, n_out, MOMENTUM))             # add dense layer
    model.add(Tanh())                                   # add activation
    start_idx = len(HIDDEN_LAYERS) - len(DROPOUT_RATES) # find the starting layer to begin applying dropout
    if i >= start_idx:                                  # shift dropouts to later layers
        rate_idx = i - start_idx                        # find the corresponding dropout idx
        model.add(Dropout(DROPOUT_RATES[rate_idx]))     # apply dropout to make neurons dead to avoid overfitting
model.add(Dense(HIDDEN_LAYERS[-1], CLASSES, MOMENTUM))  # output -> CLASSES (no dropout before output)
model.add(Softmax())
model.set_loss(CCE())

# Training loop
train_losses, val_losses = [], []
train_acc_history, val_acc_history = [], []

for epoch in range(EPOCHS):
    # mini-batch training - iterate over shuffled batches each epoch
    model.training = True                   # enable dropout during training
    epoch_losses = []
    for X_batch, y_batch_onehot in batch_generator(X_train, y_train_onehot, BATCH_SIZE):
        y_batch_pred = model.forward(X_batch)
        batch_loss = model.loss.forward(y_batch_pred, y_batch_onehot)
        epoch_losses.append(batch_loss)
        model.backward(y_batch_pred, y_batch_onehot, LR)

    # compute epoch-level training loss on full train set (no weight update)
    model.training = False                  # disable dropout for evaluation
    y_pred = model.forward(X_train)
    loss = model.loss.forward(y_pred, y_train_onehot)
    train_losses.append(loss)

    # validation loss
    y_val_pred = model.forward(X_val)
    val_loss = model.loss.forward(y_val_pred, y_val_onehot)
    val_losses.append(val_loss)

    # compute accuracies
    y_pred_class = np.argmax(y_pred, axis=1)
    y_val_class = np.argmax(y_val_pred, axis=1)
    train_acc = np.mean(y_pred_class == y_train)
    val_acc = np.mean(y_val_class == y_val)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    # print accuracy every PRINT_EVERY epochs
    if epoch % PRINT_EVERY == 0 or epoch == EPOCHS - 1:
        print(f'Epoch {epoch:>5} / {EPOCHS} | '
              f'Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | '
              f'Train Acc: {train_acc * 100:.1f}% | Val Acc: {val_acc * 100:.1f}%')

# plot decision boundary once at the end
fig, ax = plt.subplots(figsize=(6, 6))
plot_decision_boundary(model, X_train, y_train, X_val, y_val, ax=ax, grid_step=0.003,
                       title=f'Final Decision Boundary | Train Acc: {train_acc_history[-1] * 100:.1f}% | Val Acc: {val_acc_history[-1] * 100:.1f}%')
plt.show()

# Plot training & validation accuracy
plot_accuracy_curve(train_acc_history, val_acc_history)