import argparse
import glob

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import wa_constants
import wa_utils

parser = argparse.ArgumentParser()
parser.add_argument('prefix')
parser.add_argument('-d', '--dir', required=True)
parser.add_argument('-o', '--output_dir', required=True)
args = parser.parse_args()


def clustering(dir, a, prefix, eps=2, save=True):
    pca = PCA(n_components=100, svd_solver='arpack')

    c = [wa_utils.flatten_image(b) for b in a]

    x = pca.fit_transform(c)

    dbscan = DBSCAN(eps=eps, min_samples=1)
    labels = dbscan.fit_predict(x)
    for i, l in enumerate(np.unique(labels)):
        for j in [x for x, r in enumerate(labels) if l == r]:
            if save:
                cv2.imwrite(f'{dir}/{prefix}_{j}.png', a[j])
                break
    return labels


image_list = glob.glob(f'{args.dir}/*.png')

puyo_image_list = []
for file in image_list:
    print(f'processing {file}')
    im = cv2.imread(file)

    for y in range(wa_constants.FIELD_HEIGHT):
        for x in range(wa_constants.FIELD_WIDTH):
            rx = wa_constants.OFFSET_X + wa_constants.REAL_PUYO_WIDTH * x
            ry = wa_constants.OFFSET_Y + wa_constants.REAL_PUYO_HEIGHT * y
            cropped = im[ry: ry + wa_constants.REAL_PUYO_HEIGHT, rx: rx + wa_constants.REAL_PUYO_WIDTH]
            puyo_image_list.append(cropped)

clustering(args.output_dir, puyo_image_list, args.prefix)
