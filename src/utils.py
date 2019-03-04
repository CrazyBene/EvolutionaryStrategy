import numpy as np
from skimage.transform import resize
from skimage.measure import block_reduce

def rgb2gray(screen):
    return np.dot(screen[..., :3], [0.299, 0.587, 0.114])

def rescale_image(image):
    return block_reduce(image, (3, 3), cval=0)

def normalize_image(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] /= 255
    return image