# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 04:09:59 2020

@author: ASUS
"""

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import argparse
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
from PIL import ImageFile
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--1_inceptionoutputloss", type=str, default="1_inceptionoutputloss.png",help="path to output loss plot")
ap.add_argument("-p2", "--2_inceptionaccuracy", type=str, default="2_inceptionaccuracy.png",help="path to output accuracy plot")
ap.add_argument("-p3", "--3_vggconfusionmatrix", type=str, default="3_vggconfusionmatrix.png",help="confusion matrix plot")
args = vars(ap.parse_args())

def image_gen_w_aug(train_parent_directory, test_parent_directory):
    
    train_datagen = ImageDataGenerator(rescale=1/255,
                                      rotation_range = 30,  
                                      zoom_range = 0.2, 
                                      width_shift_range=0.1,  
                                      height_shift_range=0.1,
                                      validation_split = 0.15)
    
    test_datagen = ImageDataGenerator(rescale=1/255)
    
    train_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                       target_size = (75,75),
                                                       batch_size = 214,
                                                       class_mode = 'categorical',
                                                       subset='training')
    
    val_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                          target_size = (75,75),
                                                          batch_size = 37,
                                                          class_mode = 'categorical',
                                                          subset = 'validation')
    
    test_generator = test_datagen.flow_from_directory(test_parent_directory,
                                                     target_size=(75,75),
                                                     batch_size = 37,
                                                     class_mode = 'categorical')
    
    return train_generator, val_generator, test_generator


def model_output_for_TL (pre_trained_model, last_output):

    x = Flatten()(last_output)
    
    # Dense hidden layer
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output neuron. 
    x = Dense(3, activation='softmax')(x)
    
    model = Model(pre_trained_model.input, x)
    
    return model


train_dir = os.path.join('C:/Python/xyz/XYZdataset/train/')
test_dir = os.path.join('C:/Python/xyz/XYZdataset/test/')

train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, test_dir)

# Creating pre_trained_model using InceptionV3 with imagenetweight
pre_trained_model = InceptionV3(input_shape = (75, 75, 3), 
                                include_top = False, 
                                weights = 'imagenet')

pre_trained_model.trainable = False

last_layer = pre_trained_model.get_layer('mixed3')
last_output = last_layer.output

model_TL = model_output_for_TL(pre_trained_model, last_output)
model_TL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Building final model
history_TL = model_TL.fit(
      train_generator,
      steps_per_epoch=10,  
      epochs=15,
      verbose=1,
      validation_data = validation_generator)

tf.keras.models.save_model(model_TL,'inceptionmodel_XYZ.hdf5') #Saving hdf5 model    

#--------------------Data Visualisaton--------------------
#####Loss of both training and validation dataset graph#####
N = 15 #Set number of epochs as 15

# Plot graphs according to ggplot style and create figure object for loss graph
plt.style.use("ggplot")
plt.figure()
     
# loss values based on training and validation dataset from final model
loss_train = history_TL.history['loss']
loss_val = history_TL.history['val_loss']

# Plot loss_train and loss_val against no.of epochs (Green-train,Blue-validation) 
plt.plot(np.arange(0,N),loss_train,'g',label = 'Training Loss') 
plt.plot(np.arange(0,N),loss_val,'b',label = 'Validation Loss')

# Set the x label, y label,title and display legend on graph
plt.title('Training and Validation Loss on Dataset')
plt.xlabel('Epochs #')
plt.ylabel('Loss')
plt.legend()

# Save the graph as an image file named plot
plt.savefig(args["1_inceptionoutputloss"])


#----------Accuracy of both training and validation dataset graph----------

# Plot graphs according to ggplot style and create figure object for loss graph
plt.style.use("ggplot") 
plt.figure()

# Accuracy values based on training and validation dataset from final model
acc_train = history_TL.history['accuracy']
acc_val = history_TL.history['val_accuracy']

# Plot acc_train and acc_val against no.of epochs (Green-train,Blue-validation)
plt.plot(np.arange(0,N),acc_train,'g',label = 'Training Accuracy')
plt.plot(np.arange(0,N),acc_val,'b',label = 'Validation Accuracy')

# Set the x label, y label,title and display legend on graph
plt.title('Training and Validation Accuracy on Dataset')
plt.xlabel('Epochs #')
plt.ylabel('Accuracy')
plt.legend()

# Save the graph as an image file named plot2
plt.savefig(args["2_inceptionaccuracy"])


#----------Function to form confusion Matrix----------
def evaluate(model):
    #Add image-augmentation parameters to ImageDataGenerator
    img_generator = ImageDataGenerator(validation_split=0.2, 
                                  rescale = 1./255,
                                  rotation_range=20,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  vertical_flip = True
                                  )
    # Flow validation images in batches of 32 using validation_generator
    test_generator = img_generator.flow_from_directory(test_dir,
                                                          target_size = (75,75),
                                                          batch_size = 32,
                                                          class_mode = 'categorical',
                                                          subset = 'validation')
    
    # Set number of batch size and number of test samples
    batch_size = 32
    num_of_test_samples = len(test_generator.filenames)
    
    # Predict_generator: Reads data from directory containing the train set and return predictions 
    # (generator yielding batches of input samples, total number of steps)
    Y_pred = model.predict_generator(test_generator, num_of_test_samples // batch_size+1)
    
    # Return the maximum value from Y_pred along the columns
    y_pred = np.argmax(Y_pred, axis=1)

    # Form the confusion matrix y_true (actual) against y_predict
    cm=confusion_matrix(test_generator.classes, y_pred,normalize='all')
    
    # Created a dataframe called df_cm with the confusion matrix as data
    # Set indexes and columns names as classes x, y and z
    classes = ['x','y','z']
    df_cm = pd.DataFrame(cm,index=classes,columns=classes)
    
    # Set the figure size and title of graph
    plt.figure(figsize=(10,7))
    plt.title("Confusion Matrix")
    
    # Plot the graph with seaborn
    cm_plot = sn.heatmap(df_cm,annot=True)
    
    # Set x, y label of the confusion matrix graph and save the plot as plot3
    cm_plot.set(xlabel = "Actual Value",ylabel="Predicted Value")
    cm_plot.figure.savefig(args["3_vggconfusionmatrix"])


#----------Calling of function with model as parameters----------
evaluate(model_TL)
