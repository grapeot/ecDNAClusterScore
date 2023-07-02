# Test sklearn GMM
import numpy as np
import matplotlib.pyplot as pp
from PIL import Image
from tqdm import tqdm
from math import atan2, pi, exp
from sklearn.mixture import GaussianMixture
from time import time
from os.path import join, basename
from os import listdir
from glob import glob
from math import sqrt, floor
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import argparse
from GMM import visualize3d, gmmScore


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates the ecAND cluster score of the given image, and outputs a visualization figure.')
    parser.add_argument('--index', '-i', help='Channel of the image. Use 0 for R, and 1 for G.', type=int, default=0)
    parser.add_argument('--gaussians', '-n', help='Number of the Gaussians, i.e. the n in the document.', default=6)
    parser.add_argument('--threshold', '-t', help='The threshold on the PDF, i.e. the t in the email. Note the t in the document is actually this t multiplies n.', default=0.5)
    parser.add_argument('--image', help='The input image', required=True)
    parser.add_argument('--output', '-o', help='When specified, output a visualization to the given path', default=None)
    args = parser.parse_args()

    img = Image.open(args.image)
    score, gmm = gmmScore(img, index=args.index, n_components=args.gaussians, score_floor=args.threshold)
    if args.output is not None:
        visualize3d(gmm, img, f'{args.image} {score}', colorIndex=args.index, outfn=args.output)
    print(f'{args.image}\t{score}')
