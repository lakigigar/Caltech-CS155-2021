########################################
# CS/CNS/EE 155 2018
# Problem Set 3
#
# Author:       Andrew Kang
# Description:  Set 3 boosting helper
########################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Warning: slightly ugly code (copy-pasted functions).


####################
# DATASET FUNCTIONS
####################

def generate_dataset(N_train, N_test, n_spirals, r):
    # First, we express the dataset in polar coordinates.
    # The points will have radiuses proportional to their angles.
    # The second half of the points will be rotated by pi to create the second spiral.
    # Noise will be added such that the spirals are imperfect.
    np.random.seed(420)
    N = N_train + N_test
    thetas = np.random.rand(N) * (n_spirals * 2 * np.pi)
    radiuses = np.copy(thetas)
    radiuses += np.random.rand(N) * r
    thetas[int(N/2):] += np.pi

    # Convert the polar coordinates to Cartesian coordinates.
    data = np.zeros((len(thetas), 2))
    data[:, 0] = radiuses * np.cos(thetas)
    data[:, 1] = radiuses * np.sin(thetas)

    # Label the points. Blue will be +1 and red will be -1.
    labels = np.ones(N)
    labels[int(N/2):] *= -1
    data = np.column_stack((data, labels))

    # Shuffle the dataset and split it into training and testing datasets.
    np.random.shuffle(data)
    data_train, data_test = data[:N_train, :], data[N_train:, :]

    # Split the training and testing datasets into X and y.
    X_train, Y_train = data_train[:, :2], data_train[:, 2]
    X_test, Y_test = data_test[:, :2], data_test[:, 2]

    return (X_train, Y_train), (X_test, Y_test)


####################
# PLOTTING FUNCTIONS
####################

def visualize_dataset(X, Y, title):
    # Set colormap such that blue is positive and red is negative.
    plt.close('all')
    plt.set_cmap('bwr')
    plt.figure(figsize=(6, 5))

    # Plot data.
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=10)
    plt.title(title)
    plt.colorbar()
    plt.show()

def visualize_predictions(X, Y, Y_pred, title):
    # Set colormap such that red is positive and blue is negative.
    plt.close('all')
    plt.set_cmap('bwr')
    plt.figure(figsize=(6, 5))

    # Plot data.
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=10, linewidth=np.abs(Y - Y_pred), edgecolors='black')
    plt.title(title)
    plt.colorbar()
    plt.show()

def visualize_loss_curves_gb(model, X_train, Y_train, X_test, Y_test):
    plt.close('all')

    Y_train_pred = np.zeros_like(Y_train)
    Y_test_pred = np.zeros_like(Y_test)
    losses_train = []
    losses_test = []

    # For each added classifier, store the new training and test losses.
    for clf in model.clfs:
        Y_train_pred += clf.predict(X_train)
        Y_test_pred += clf.predict(X_test)

        losses_train.append(len(np.where(np.sign(Y_train_pred) != Y_train)[0]) / len(Y_train_pred))
        losses_test.append(len(np.where(np.sign(Y_test_pred) != Y_test)[0]) / len(Y_test_pred))

    # Plot the losses across n_clfs.
    plt.plot(np.arange(1, model.n_clfs + 1), losses_train)
    plt.plot(np.arange(1, model.n_clfs + 1), losses_test)
    plt.title('Loss vs. n_clfs')
    plt.legend(['Training loss', 'Test loss'])
    plt.xlabel('n_clfs')
    plt.ylabel('LOss')
    plt.show()

def visualize_loss_curves_ab(model, X_train, Y_train, X_test, Y_test):
    plt.close('all')

    Y_train_pred = np.zeros_like(Y_train)
    Y_test_pred = np.zeros_like(Y_test)
    losses_train = []
    losses_test = []

    # For each added classifier, store the new training and test losses.
    for i, clf in enumerate(model.clfs):
        Y_train_pred += model.coefs[i] * clf.predict(X_train)
        Y_test_pred += model.coefs[i] * clf.predict(X_test)

        losses_train.append(len(np.where(np.sign(Y_train_pred) != Y_train)[0]) / len(Y_train_pred))
        losses_test.append(len(np.where(np.sign(Y_test_pred) != Y_test)[0]) / len(Y_test_pred))

    # Plot the losses across n_clfs.
    plt.plot(np.arange(1, model.n_clfs + 1), losses_train)
    plt.plot(np.arange(1, model.n_clfs + 1), losses_test)
    plt.title('Loss vs. n_clfs')
    plt.legend(['Training loss', 'Test loss'])
    plt.xlabel('n_clfs')
    plt.ylabel('LOss')
    plt.show()


####################
# BOOSTING SUITE FUNCTIONS
####################

def gb_suite(Boosting, n_clfs, X_train, Y_train, X_test, Y_test):
    # Fit a boosting model.
    model = Boosting(n_clfs=n_clfs)
    model.fit(X_train, Y_train)

    # Report the prediction plots and losses.
    visualize_predictions(X_train, Y_train, model.predict(X_train), 'Training dataset predictions')
    print('Training loss: %f' % model.loss(X_train, Y_train))

    visualize_predictions(X_test, Y_test, model.predict(X_test), 'Test dataset predictions')
    print('Test loss: %f' % model.loss(X_test, Y_test))

    return model

# Same as above but also returns D (dataset weights across all iterations).
def ab_suite(AdaBoost, n_clfs, X_train, Y_train, X_test, Y_test):
    # Fit a boosting model.
    model = AdaBoost(n_clfs=n_clfs)
    D = model.fit(X_train, Y_train)

    # Report the prediction plots and losses.
    visualize_predictions(X_train, Y_train, model.predict(X_train), 'Training dataset predictions')
    print('Training loss: %f' % model.loss(X_train, Y_train))

    visualize_predictions(X_test, Y_test, model.predict(X_test), 'Test dataset predictions')
    print('Test loss: %f' % model.loss(X_test, Y_test))

    return model, D


####################
# BOOSTING ANIMATION FUNCTIONS
####################

def animate_gb(model, X, Y, title):
    size = 10
    step = 5
    Y_clfs = np.zeros((len(X), len(model.clfs)))

    # Store predictions from each DT weak regressor in a different column.
    for i, clf in enumerate(model.clfs):
        Y_curr = clf.predict(X)
        Y_clfs[:, i] = Y_curr
    
    # Initialize graph to animate on.
    fig = plt.figure(figsize=(6, 5))
    scat = plt.scatter(X[:, 0], X[:, 1], c=Y, s=size)
    plt.colorbar()
    plt.title(title)

    # Define frame animation function.
    def animate(i, Y_clfs, scat):
        scat.set_array(np.sum(Y_clfs[:, :step*i], axis=1).T)
        return scat,

    # Animate!
    print('\nAnimating...')
    anim = animation.FuncAnimation(fig, animate, frames=np.arange(int(model.n_clfs / step)), fargs=(Y_clfs, scat))

    return anim

# Same as above but also shows D (dataset weights across all iterations).
def animate_ab(model, X, Y, D, title):
    size = 10
    step = 5
    Y_clfs = np.zeros((len(X), len(model.clfs)))

    # Store predictions from each DT weak regressor in a different column.
    for i, clf in enumerate(model.clfs):
        Y_curr = model.coefs[i] * clf.predict(X)
        Y_clfs[:, i] = Y_curr
    
    # Initialize graph to animate on.
    fig = plt.figure(figsize=(6, 5))
    scat = plt.scatter(X[:, 0], X[:, 1], c=Y, s=size)
    plt.colorbar()
    plt.title(title)

    # Define frame animation function.
    def animate(i, Y_clfs, scat):
        scat.set_array(np.sum(Y_clfs[:, :step*i], axis=1).T)
        scat.set_sizes(20000 * D[:, step*i].T)
        return scat,

    # Animate!
    print('\nAnimating...')
    anim = animation.FuncAnimation(fig, animate, frames=np.arange(int(model.n_clfs / step)), fargs=(Y_clfs, scat))

    return anim
