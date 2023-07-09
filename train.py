import matplotlib.pyplot as plt
import pandas as pd                                     # Data analysis and manipultion tool
import numpy as np                                      # Fundamental package for linear algebra and multidimensional arrays
import tensorflow as tf                                 # Deep Learning Tool                                              # OS module in Python provides a way of using operating system dependent functionality
import cv2                                              # Library for image processing
from sklearn.model_selection import train_test_split    # For splitting the data into train and validation set
from tensorflow.keras import datasets, layers, models   #Tensorflow for the model
from sklearn.metrics import accuracy_score
import shutil #File management library
from tqdm.notebook import tqdm
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,Add, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Train Eye CNN Model')
parser.add_argument('--zip_file', type = str, help = 'path to Training ZIP file')
parser.add_argument('--no_epochs', type = int, default=30, help = 'No of epochs')
parser.add_argument('--image_size', type = int, default=100, help = 'Standard Image Size')
args = parser.parse_args()

#Function to unpack the data from our zip file
def unpack_zip_file(zip_file):
    shutil.unpack_archive( zip_file, 'eye_gender_data/')
  
#Func to generate our dataset
def generate_dataset(image_size = 100):
    #Here, we load the labels from their csv file
    labels = pd.read_csv("eye_gender_data/eye_gender_data/Training_set.csv")
    #For every label, we extract the file_path to that particular image
    file_paths = [[fname, 'eye_gender_data/eye_gender_data/train/' + fname] for fname in labels['filename']]
    images = pd.DataFrame(file_paths, columns=['filename', 'filepaths'])
    data = pd.merge(images, labels, how = 'inner', on = 'filename')

    #To avoid variability in image size, as we're using a basic CNN model, we fix the image size to 100. 
    #This is what works best for this dataset. Other sizes can be taken too.
    #However, once the size is fixed, it should be used for all the training and test data. Our CNN can't accept different image sizes

    #Resize every image to our pre-set image size
    X = np.array([cv2.resize(cv2.imread(data['filepaths'][i], cv2.IMREAD_GRAYSCALE),(image_size,image_size)).reshape(image_size,image_size,1) for i in range(len(images))])
    #Mapping labels to binary
    #0 for male and 1 for female
    Y = np.array([0.0 if (data['label'][i]) == 'male' else 1.0 for i in range(len(images))])
    return X,Y

#Here, we define our Data Augmentation
def augment(inputs: Tensor) -> Tensor:
    out = layers.experimental.preprocessing.RandomFlip("horizontal")(inputs)
    out = layers.experimental.preprocessing.RandomRotation(0.1)(out)
    out = layers.experimental.preprocessing.RandomZoom(0.5)(out)
    out = layers.experimental.preprocessing.RandomCrop(image_size,image_size)(out)
    return out

#We define relu_bn as a combo of ReLU + BatchNorm layers
def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

#We define a residual_block which uses a skip connection here
def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

#Here, we combine our final model architecture
def create_model():              
    inputs = Input(shape=(image_size, image_size, 1))
    
    num_filters = 64
    t = inputs
    t = augment(t)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    
    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
            t = Dropout(0.3)(t)
        num_filters *= 2
    
    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    outputs = Dense(64, activation='relu')(t)
    outputs = Dense(1, activation='sigmoid')(outputs)

    model = Model(inputs, outputs)
    
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),metrics=['accuracy'])    

    return model

def split(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    return x_train, x_test, y_train, y_test

def train(x_train, y_train, model, no_epochs = 30, batch_size = 64):
    model.fit(x_train, y_train, epochs=no_epochs, batch_size=batch_size)
    
def predict(model, x_test, y_test):
    preds = model.predict(x_test)
    preds = [0.0 if preds[i] <=0.5 else 1.0 for i in range(len(preds))]
    acc = accuracy_score(y_test,preds)
    return preds, acc

if __name__ == '__main__':
    zip_file = args.zip_file
    no_epochs = args.no_epochs
    image_size = args.image_size
    
    Path('eye_gender_data').mkdir(parents=True, exist_ok=True)
    
    unpack_zip_file(zip_file)
    X, y = generate_dataset(image_size)
    
    model = create_model()
    x_train, x_test, y_train, y_test = split(X, y)
    train(x_train, y_train, model, no_epochs)
    
    preds, acc = predict(model, x_test, y_test)
    print("Accuracy on test set = {:.2f} %".format(acc))
    
    model.save('my_trained_model')