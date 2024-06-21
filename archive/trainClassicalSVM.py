from pennylane import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pennylane as qml
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import KernelCenterer


def _make_circular_data(num_sectors):
    """Generate datapoints arranged in an even circle."""
    center_indices = np.array(range(0, num_sectors))
    sector_angle = 2 * np.pi / num_sectors
    angles = (center_indices + 0.5) * sector_angle
    x = 0.7 * np.cos(angles)
    y = 0.7 * np.sin(angles)
    labels = 2 * np.remainder(np.floor_divide(angles, sector_angle), 2) - 1

    return x, y, labels


def make_double_cake_data(num_sectors):
    x1, y1, labels1 = _make_circular_data(num_sectors)
    x2, y2, labels2 = _make_circular_data(num_sectors)

    # x and y coordinates of the datapoints
    x = np.hstack([x1, 0.5 * x2])
    y = np.hstack([y1, 0.5 * y2])

    # Canonical form of dataset
    X = np.vstack([x, y]).T

    labels = np.hstack([labels1, -1 * labels2])

    # Canonical form of labels
    Y = labels.astype(int)

    return X, Y


def plot_double_cake_data(X, Y, ax, num_sectors=None):
    """Plot double cake data and corresponding sectors."""
    x, y = X.T
    cmap = mpl.colors.ListedColormap(["#FF0000", "#0000FF"])
    ax.scatter(x, y, c=Y, cmap=cmap, s=25, marker="s")

    if num_sectors is not None:
        sector_angle = 360 / num_sectors
        for i in range(num_sectors):
            color = ["#FF0000", "#0000FF"][(i % 2)]
            other_color = ["#FF0000", "#0000FF"][((i + 1) % 2)]
            ax.add_artist(
                mpl.patches.Wedge(
                    (0, 0),
                    1,
                    i * sector_angle,
                    (i + 1) * sector_angle,
                    lw=0,
                    color=color,
                    alpha=0.1,
                    width=0.5,
                )
            )
            ax.add_artist(
                mpl.patches.Wedge(
                    (0, 0),
                    0.5,
                    i * sector_angle,
                    (i + 1) * sector_angle,
                    lw=0,
                    color=other_color,
                    alpha=0.1,
                )
            )
            ax.set_xlim(-1, 1)

    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    return ax

def powerFaultData():

    powerData = pd.read_csv('Testdata.csv')
    
    X = []
    Y = []

    for i in powerData.iterrows():
        X.append([i[1][p] for p in range(0, len(i[1]))])
        Y.append(int(i[1][6]))

    Y = np.asarray(Y).astype(int)

    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.20, shuffle=True, random_state=42)
    xTrain = preprocessing.normalize(xTrain)
    xTest = preprocessing.normalize(xTest)

    return xTrain, yTrain, xTest, yTest


def kernel_alignment(X_source, X_target, kernel='linear'):
    # Calculate kernel matrices for source and target domains
    K_source = pairwise_kernels(X_source, metric=kernel)
    K_target = pairwise_kernels(X_target, metric=kernel)
    
    # Center the kernel matrices
    centerer = KernelCenterer()
    K_source_centered = centerer.fit_transform(K_source)
    K_target_centered = centerer.transform(K_target)
    
    # Compute alignment measure
    alignment = np.trace(np.dot(K_source_centered, K_target_centered.T))
    
    return alignment, K_source_centered, K_target_centered

X, Y, Xtest, Ytest = powerFaultData()

alignment_score, aligned_K_source, aligned_K_target = kernel_alignment(X, X)
alignment_score_test, aligned_K_source_test, aligned_K_target_test = kernel_alignment(Xtest, Xtest)

svm = SVC(kernel='precomputed')
svm.fit(aligned_K_target, Y)

accuracy = svm.score(aligned_K_target_test, Ytest)
print("Accuracy:", accuracy)