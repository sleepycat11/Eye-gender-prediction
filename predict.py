import matplotlib.pyplot as plt
import pandas as pd                                     # Data analysis and manipultion tool
import numpy as np                                      # Fundamental package for linear algebra and multidimensional arrays
import cv2                                              # Library for image processing
import shutil
from tensorflow import keras
import argparse

parser = argparse.ArgumentParser(description='Predict CNN Results')
parser.add_argument('--zip_file', type = str, help = 'path to Test Data ZIP file')
parser.add_argument('--model_path', type = str, default= 'eye_cnn_model', help = 'Path to model')
parser.add_argument('--image_size', type = int, default=100, help = 'Standard Image Size')
parser.add_argument('--output_path', type = str, default= './', help = 'Path to output')
args = parser.parse_args()

#Function to unpack the data from our zip file
def unpack_zip_file(zip_file):
    shutil.unpack_archive( zip_file, 'eye_gender_data/')
  
#Func to generate our dataset
def generate_dataset(image_size = 100):
    #Here, we load the file paths from their csv file
    labels = pd.read_csv("eye_gender_data/eye_gender_data/Testing_set.csv")
    #For every label, we extract the file_path to that particular image
    file_paths = [[fname, 'eye_gender_data/eye_gender_data/train/' + fname] for fname in labels['filename']]
    images = pd.DataFrame(file_paths, columns=['filename', 'filepaths'])

    #To avoid variability in image size, as we're using a basic CNN model, we fix the image size to 100. 
    #This is what works best for this dataset. Other sizes can be taken too.
    #However, once the size is fixed, it should be used for all the training and test data. Our CNN can't accept different image sizes

    #Resize every image to our pre-set image size
    X = np.array([cv2.resize(cv2.imread(images['filepaths'][i], cv2.IMREAD_GRAYSCALE),(image_size,image_size)).reshape(image_size,image_size,1) for i in range(len(images))])
    return X

def predict(model, x_test):
    preds = model.predict(x_test)
    preds = [0.0 if preds[i] <=0.5 else 1.0 for i in range(len(preds))] #Convert predictions to binary
    return preds

if __name__ == '__main__':
    
    zip_file = args.zip_file
    model_path = args.model_path
    image_size = args.image_size
    output_path = args.output_path
    
    
    unpack_zip_file(zip_file)
    X = generate_dataset(image_size)
    model = keras.models.load_model(model_path) #We load our model
    preds = predict(model, X) #We run predictions
    
    df = pd.DataFrame(preds, columns = ['Predictions'])
    df.to_csv(output_path+'Eye_Gender_Predictions.csv', index = False)
    