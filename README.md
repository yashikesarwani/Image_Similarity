# Image_Similarity
Image Similarity using Transfer Learning

I did this using the technique Transfer Learning. In this task, we find similar images
by dissecting the weights of image-object classifier VGG and using it to extract 
feature vectors from the given dataset to find the similar images to an input image.


![alt text](https://github.com/yashikesarwani/Image_Similarity/blob/master/output/vgg19/vgg19_retrieval_0.png)


Steps for finding similar images:
 •
• • • •
•
From the given dataset, put all the images into the training folder inside the dataset directory and then choosing randomly an image from the training folder images to select it as a query image and putting it in a testing directory inside the dataset directory.
Load the trained VGG model and remove the last layers.
Convert the training images into feature vectors using dissected vector.
Perform inference on our image vectors for the generation of flattened embeddings
Compute the similarities between our images feature vectors using an inner product cosine similarity. We used KNN Algorithm.
For the query image selected randomly inside the testing folder of dataset, the top k=9 similar images are found as per the rank of the images based on the top similarity scores.
