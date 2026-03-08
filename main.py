# imports
import numpy as np
import matplotlib.pyplot as plt

# create synthetic spiral data
def spiral_data(points: int, classes: int, radius: float, noise: float) -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, radius, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*noise
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

# plot spiral data
X, y = spiral_data(100, 3, 1.0, 0.1)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Spiral Data')
plt.show()