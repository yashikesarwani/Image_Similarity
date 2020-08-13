"""
Reading and Saving Image

"""
import os
import skimage.io
from multiprocessing import Pool

# Reading the image
def read_img(filePath):
    return skimage.io.imread(filePath, as_gray=False)

# Reading images with common extensions from a given directory
def read_imgs_dir(dirPath, extensions, parallel=True):
    args = [os.path.join(dirPath, filename)
            for filename in os.listdir(dirPath)
            if any(filename.lower().endswith(ext) for ext in extensions)]
    if parallel:
        pool = Pool()
        imgs = pool.map(read_img, args)
        pool.close()
        pool.join()
    else:
        imgs = [read_img(arg) for arg in args]
    return imgs

# Saving image to a file
def save_img(filePath, img):
    skimage.io.imsave(filePath, img)