import cv2
from tqdm import tqdm
import glob
import numpy as np


def compute_img_mean_std(path="data/ham10000/images"):
    img_h, img_w = 224, 224
    imgs = []
    means, stdevs = [], []

    image_paths = [f for f in glob.glob(f"{path}/*.jpg")]

    for i in tqdm(range(len(image_paths))):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.0

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means, stdevs


if __name__ == "__main__":
    compute_img_mean_std(path="data/ham10000/images")
