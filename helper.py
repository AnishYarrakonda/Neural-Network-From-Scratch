#imports
import matplotlib.pyplot as plt
import numpy as np

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


# helper function: yields mini-batches
def batch_generator(X, y, batch_size):
    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


# helper function: plots the loss curve
def plot_accuracy_curve(train_acc, test_acc=None):
    """
    Plots training and optional testing accuracy curve,
    highlighting the best testing accuracy with a green vertical line.

    Parameters:
        train_acc: list or array of training accuracy values (0–1)
        test_acc: optional list or array of testing accuracy values (0–1)
    """
    train_acc = np.array(train_acc)
    plt.figure(figsize=(8, 5))

    # plot training accuracy
    plt.plot(train_acc, label='Train Accuracy', color='blue')

    if test_acc is not None:
        test_acc = np.array(test_acc)
        # plot testing accuracy
        plt.plot(test_acc, label='Test Accuracy', color='orange')

        # find best testing accuracy
        best_epoch = np.argmax(test_acc)
        best_acc = test_acc[best_epoch]

        # vertical line at best testing accuracy
        plt.axvline(x=best_epoch, color='green', linestyle='--',
                    label=f'Best Test Acc Epoch {best_epoch + 1} ({best_acc * 100:.2f}%)')
        plt.scatter(best_epoch, best_acc, color='green', zorder=5)

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


# helper function: plots decision boundary
def plot_decision_boundary(model, X_train, y_train, X_val=None, y_val=None, grid_step=0.01, ax=None, title=None):
    """
    Plots the decision boundary of a trained model.

    Parameters:
        model: NNModel instance
        X_train, y_train: training data
        X_val, y_val: optional validation data
        grid_step: step size for meshgrid
        ax: optional matplotlib Axes object
        title: optional plot title
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # create meshgrid
    x_min, x_max = X_train[:, 0].min() - 0.1, X_train[:, 0].max() + 0.1
    y_min, y_max = X_train[:, 1].min() - 0.1, X_train[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))

    # predict class for each point in meshgrid
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.forward(grid)
    Z = np.argmax(probs, axis=1).reshape(xx.shape)

    # plot contour
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

    # plot training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', label='Train')

    # plot validation points if given
    if X_val is not None and y_val is not None:
        ax.scatter(X_val[:, 0], X_val[:, 1], c=y_val, edgecolor='k', marker='o', label='Val')

    # labels, title, legend
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    if title is not None:
        ax.set_title(title)
    ax.legend(loc='upper right')

    return ax