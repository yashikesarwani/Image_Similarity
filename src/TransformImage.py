"""

 Transformation of Images

"""
from multiprocessing import Pool
from skimage.transform import resize

# Applying transformations to multiple images
def apply_transformer(imgs, transformer, parallel=True):
    if parallel:
        pool = Pool()
        imgs_transform = pool.map(transformer, [img for img in imgs])
        pool.close()
        pool.join()
    else:
        imgs_transform = [transformer(img) for img in imgs]
    return imgs_transform

# Normalizing image data [0, 255] -> [0.0, 1.0]
def normalize_img(img):
    return img / 255.

# Resizing of image
def resize_img(img, shape_resized):
    img_resized = resize(img, shape_resized,
                         anti_aliasing=True,
                         preserve_range=True)
    assert img_resized.shape == shape_resized
    return img_resized

# Flattening of image
def flatten_img(img):
    return img.flatten("C")