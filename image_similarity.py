"""

 image_similarity.py  

 Performing image similarity using transfer learning on a pre-trained
 VGG image classifier. Plotting k=9 the most similar images to the
 query images, as well as the t-SNE visualizations.

"""

#Importing libaraies
import os
import random
import glob
import cv2
import numpy as np
import shutil 
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from src.InputOutput import read_imgs_dir
from src.TransformImage import apply_transformer
from src.TransformImage import resize_img, normalize_img
from src.PlotImage import plot_query_retrieval, plot_tsne, plot_reconstructions


modelName = "vgg19" # model used for transfer learning
parallel = True  # use multicore processing
k = 9

# Making paths for the training images and testing (query) image
dataTrainDir = os.path.join(os.getcwd(), "dataset", "training")
dataTestDir = os.path.join(os.getcwd(), "dataset", "testing")

# Making test directory by removing the existing one and a new one if it does not exist
if os.path.exists(dataTestDir):
    shutil.rmtree(dataTestDir)
os.makedirs(dataTestDir)

# Reading an image from training images directory randomly to be taken as input query image
data_path = os.path.join(dataTrainDir,'*jpg') 
files = glob.glob(data_path) 
d=random.choices(files)
img = cv2.imread(d[0], 1) 


#write that image to the test directory  
cv2.imwrite(os.path.join(dataTestDir, 'queryimage.jpg'), img)
cv2.waitKey(0)

# Making Output path
outDir = os.path.join(os.getcwd(), "output", modelName)
if os.path.exists(outDir):
    shutil.rmtree(outDir)
os.makedirs(outDir)    

# Reading images
extensions = [".jpg", ".jpeg"]
print("Reading training images from '{}'...".format(dataTrainDir))
imgs_train = read_imgs_dir(dataTrainDir, extensions, parallel=parallel)
print("Reading testing image from '{}'...".format(dataTestDir))
imgs_test = read_imgs_dir(dataTestDir, extensions, parallel=parallel)
shape_img = imgs_train[0].shape
print("Image shape = {}".format(shape_img))

# Loading pre-trained VGG19 model + higher level layers
print("Loading VGG19 pre-trained model-")
model = tf.keras.applications.VGG19(weights='imagenet', include_top=False,
                                    input_shape=shape_img)
model.summary()

shape_img_resize = tuple([int(x) for x in model.input.shape[1:]])
input_shape_model = tuple([int(x) for x in model.input.shape[1:]])
output_shape_model = tuple([int(x) for x in model.output.shape[1:]])
n_epochs = None

# Printing model info
print("Input_shape_model = {}".format(input_shape_model))
print("output_shape_model = {}".format(output_shape_model))

# Applying transformations to all images
class ImageTransformer(object):

    def __init__(self, shape_resize):
        self.shape_resize = shape_resize

    def __call__(self, img):
        img_transformed = resize_img(img, self.shape_resize)
        img_transformed = normalize_img(img_transformed)
        return img_transformed

transformer = ImageTransformer(shape_img_resize)
print("Applying image transformer to training images-")
imgs_train_transformed = apply_transformer(imgs_train, transformer, parallel=parallel)
print("Applying image transformer to query image-")
imgs_test_transformed = apply_transformer(imgs_test, transformer, parallel=parallel)

# Converting images to numpy array
X_train = np.array(imgs_train_transformed).reshape((-1,) + input_shape_model)
X_test = np.array(imgs_test_transformed).reshape((-1,) + input_shape_model)
print(" -> X_train.shape = {}".format(X_train.shape))
print(" -> X_test.shape = {}".format(X_test.shape))


# Creating embeddings using model
print("Inferencing embeddings using pre-trained model-")
E_train = model.predict(X_train)
E_train_flatten = E_train.reshape((-1, np.prod(output_shape_model)))
E_test = model.predict(X_test)
E_test_flatten = E_test.reshape((-1, np.prod(output_shape_model)))
print(" -> E_train.shape = {}".format(E_train.shape))
print(" -> E_test.shape = {}".format(E_test.shape))
print(" -> E_train_flatten.shape = {}".format(E_train_flatten.shape))
print(" -> E_test_flatten.shape = {}".format(E_test_flatten.shape))


# Fitting kNN model on training images
print("Fitting k-nearest-neighbour model on training images")
knn = NearestNeighbors(n_neighbors=k, metric="cosine")
knn.fit(E_train_flatten)

# Performing image retrieval on query image
print("Performing image retrieval on query image to find '{}' most similar image".format(k))
for i, emb_flatten in enumerate(E_test_flatten):
    _, indices = knn.kneighbors([emb_flatten]) # find k nearest train neighbours
    img_query = imgs_test[i] # query image
    imgs_retrieval = [imgs_train[idx] for idx in indices.flatten()] # retrieval images
    outFile = os.path.join(outDir, "{}_retrieval_{}.png".format(modelName, i))
    plot_query_retrieval(img_query, imgs_retrieval, outFile)

# Plotting t-SNE visualization
print("Visualizing t-SNE on training images")
outFile = os.path.join(outDir, "{}_tsne.png".format(modelName))
plot_tsne(E_train_flatten, imgs_train, outFile)