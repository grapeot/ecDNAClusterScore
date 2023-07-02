import numpy as np
import matplotlib.pyplot as pp
import matplotlib.pyplot as plt
from math import atan2, pi, exp
from sklearn.mixture import GaussianMixture
from time import time
from os.path import join, basename
from os import listdir
from glob import glob
from math import sqrt, floor
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# The version has thresholding. Pixels with brightness less than the threshold will not be considered.
# Perform weighted PCA, return FWHM
# img is a PIL image
def gmmScore(img, index, threshold=30.0/255, n_components=2, score_floor=0.5):
    w, h = img.size
    h2 = floor(300 * h / w)
    img = img.resize((300, h2))
    img = np.asarray(img)[:, :, index]
    img = (img.astype('float') / img.max()).astype('float')  # normalize to [0, 1]
    img[img < threshold] = 0
    h, w = img.shape
    ys, xs = np.nonzero(img)
    data = np.asarray([[x / w, y / h] for x, y in zip(xs, ys)])
    
    gmm = GaussianMixture(random_state=0, n_init=10, n_components=n_components, covariance_type='full', max_iter=10000, tol=0.000001)
    gmm.fit(data)
        
    # V2
    X = np.arange(0, w)
    Y = np.arange(0, h)
    X, Y = np.meshgrid(X, Y)
    XY = np.stack((X / w, Y / h), axis=2)
    Z = X.copy().astype('float')
    for y in range(XY.shape[0]):
        Z[y, :] = np.exp(gmm.score_samples(XY[y, :, :]))
    maxz = Z.max()
    Z[Z < n_components * score_floor] = 0
    Z[Z < maxz / 2] = 0
    score = Z.sum() / w / h
    return score, gmm


def visualize3d(gmm, img, title='', colorIndex=2, outfn='out.jpg'):
    img = np.asarray(img)
    h, w, c = img.shape
    X = np.arange(0, w)
    Y = np.arange(0, h)
    X, Y = np.meshgrid(X, Y)
    XY = np.stack((X / w, Y / h), axis=2)
    Z = X.copy().astype('float')
    for y in range(XY.shape[0]):
        Z[y, :] = np.exp(gmm.score_samples(XY[y, :, :]))
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca(projection='3d')
#     ax.axis('off')
    ax.contourf(X, Y, img[:, :, 2] / 255.0, zdir='z', origin='lower', offset=-0.5, cmap='Reds' if colorIndex == 0 else 'Greens')
    ax.plot_surface(X, Y, Z, alpha=0.2, cmap=cm.coolwarm)
    ax.contourf(X, Y, Z, zdir='z', origin='lower', offset=-0.5, cmap=cm.coolwarm, alpha=0.3)
    ax.set_title(title)
    pp.savefig(outfn, bbox_inches='tight')
