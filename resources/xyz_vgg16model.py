import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
import argparse
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="1_vggoutputloss.png",help="path to output loss plot")
ap.add_argument("-p2", "--plot2", type=str, default="2_vggaccuracy.png",help="path to output accuracy plot")
ap.add_argument("-p3", "--plot3", type=str, default="3_vggconfusionmatrix.png",help="confusion matrix plot")
args = vars(ap.parse_args())

train_dir = 'C:/Python/xyz/XYZdataset/train' #set file location 
test_dir = 'C:/Python/xyz/XYZdataset/test'

X_files = os.listdir(os.path.join(train_dir,'X'))
Y_files = os.listdir(os.path.join(train_dir,'Y'))
Z_files = os.listdir(os.path.join(train_dir,'Z'))

print('total training x images:', len(X_files))
print('total training y images:', len(Y_files))
print('total training z images:', len(Z_files))

pic_index = 100

next_x = [os.path.join(train_dir, 'X', fname) 
                for fname in X_files[pic_index-1:pic_index]]
next_y = [os.path.join(train_dir, 'Y', fname) 
                for fname in Y_files[pic_index-1:pic_index]]
next_z = [os.path.join(train_dir, 'Z', fname) 
                for fname in Z_files[pic_index-1:pic_index]]

f, axarr = plt.subplots(1,3, figsize=(30,20))
for i, img_path in enumerate(next_x+next_y+next_z):
    img = mpimage.imread(img_path)
    axarr[i].imshow(img)
    axarr[i].axis('Off')

# Add image-augmentation parameters to ImageDataGenerator
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

# Flow training images in batches of 32 using train_generator 
train_generator = img_generator.flow_from_directory(
                        train_dir,
                        target_size=(224,224),
                        batch_size=32,
                        class_mode='categorical',
                        shuffle=True,
                        subset='training'
                    )

# Flow validation images in batches of 32 using validation_generator
validation_generator = img_generator.flow_from_directory(
                        test_dir,
                        target_size=(224,224),
                        batch_size=32,
                        class_mode='categorical',
                        shuffle=False,
                        subset='validation'
                    )

# Creating base model using VGG16 with imagenetweight for pre-training
base_model = VGG16(input_shape=(224,224,3),  #224 Image dimension and 3 labels
                   include_top=False,
                   weights='imagenet')

base_model.trainable = False        #to prevent overfitting, so model do not need to train all layers

#Building the last fully-connected layer
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),       # applies average pooling on the spatial dimensions until each spatial dimension is one, and leaves other dimensions unchanged
    tf.keras.layers.Dense(512, activation='relu'),  # Add a fully connected layer with 512 hidden units and ReLU activation
    tf.keras.layers.Dropout(0.5),                   # Add a dropout rate of 0.5, prevent overfitting and reduce training time
    tf.keras.layers.Dense(3, activation='softmax')  # Add a final softmax layer for classification
]) 

model.summary()

#Compiling model
model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics=['accuracy']) 

#Building final model based on the training and validation sets
history = model.fit_generator(
    train_generator,  
    validation_data  = validation_generator,
    epochs = 10,  #Using 10 epochs
    verbose = 1
)
model.save('vgg16model_XYZ.hdf5')  #Saving hdf5 model    


#--------------------Data Visualisaton--------------------
#####Loss of both training and validation dataset graph#####
N = 10 #Set number of epochs as 10

# Plot graphs according to ggplot style and create figure object for loss graph
plt.style.use("ggplot")
plt.figure()
     
# loss values based on training and validation dataset from final model
loss_train = history.history['loss']
loss_val = history.history['val_loss']

# Plot loss_train and loss_val against no.of epochs (Green-train,Blue-validation) 
plt.plot(np.arange(0,N),loss_train,'g',label = 'Training Loss') 
plt.plot(np.arange(0,N),loss_val,'b',label = 'Validation Loss')

# Set the x label, y label,title and display legend on graph
plt.title('Training and Validation Loss on Dataset')
plt.xlabel('Epochs #')
plt.ylabel('Loss')
plt.legend()

# Save the graph as an image file named plot
plt.savefig(args["plot"])


#----------Accuracy of both training and validation dataset graph----------

# Plot graphs according to ggplot style and create figure object for loss graph
plt.style.use("ggplot") 
plt.figure()

# Accuracy values based on training and validation dataset from final model
acc_train = history.history['accuracy']
acc_val = history.history['val_accuracy']

# Plot acc_train and acc_val against no.of epochs (Green-train,Blue-validation)
plt.plot(np.arange(0,N),acc_train,'g',label = 'Training Accuracy')
plt.plot(np.arange(0,N),acc_val,'b',label = 'Validation Accuracy')

# Set the x label, y label,title and display legend on graph
plt.title('Training and Validation Accuracy on Dataset')
plt.xlabel('Epochs #')
plt.ylabel('Accuracy')
plt.legend()

# Save the graph as an image file named plot2
plt.savefig(args["plot2"])


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
                                                          target_size = (224,224),
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
    cm_plot.figure.savefig(args["plot3"])


#----------Calling of function with model as parameters----------
evaluate(model)
