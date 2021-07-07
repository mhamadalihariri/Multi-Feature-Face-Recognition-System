#import libraries needed for data visualization 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
from sklearn.metrics import confusion_matrix, classification_report
# import needed for data summary
import glob
import os
import pandas as pd
from pandas import DataFrame
#import required libraries
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam


img = mpimg.imread("Files/content/kaggle/train/angry/Training_3908.jpg")
imgplot = plt.imshow(img)
plt.show()


img = mpimg.imread("Files/content/kaggle/train/happy/Training_10070997.jpg")
imgplot = plt.imshow(img)
plt.show()

img = mpimg.imread("Files/content/kaggle/train/surprise/Training_10060820.jpg")
imgplot = plt.imshow(img)
plt.show()


train_dir = 'Files/content/kaggle/train'
val_dir = 'Files/content/kaggle/validation'
test_dir = 'Files/content/kaggle/test'


train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)

#generate ur training, validation and testing dataset
training_set = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

val_set = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

test_set=val_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False)


path = r'Files\content\kaggle'            

#we shall store all the file names in this list
filelist = []

for root, dirs, files in os.walk(path):
  for file in files:
    if file.endswith('.jpg'):
      filelist.append(os.path.join(root,file))
  
df_files = DataFrame (filelist,columns=['fila_path'])
print(df_files)

df_data = df_files["fila_path"].str.split("\\", n = 6, expand = True)
print(df_data)
print(len(df_data))
for i in range(len(df_data)):
    df_data[0][i]=df_data[0][i].replace("Files","")
col = ['index','Content1','Path','Folder','Category','FileName']
print(col)
print(df_data)
print(df_data.columns)
df_data.columns=col

df_data=df_data[['Folder','Category']]

df_data.groupby(["Folder"]).count()

df_data.groupby(["Folder", "Category"]).size()

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

emotion_model.add(MaxPooling2D(pool_size=(2, 2)))

emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())

emotion_model.add(Dense(1024, activation='relu',))
emotion_model.add(Dropout(0.25))
emotion_model.add(Dense(5, activation='softmax'))

#now let's see a summary of our model
emotion_model.summary()
emotion_model.save


emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001),metrics=['accuracy'])

emotion_model_info = emotion_model.fit(
        training_set,
        epochs=25,
        validation_data=val_set)

emotion_model.save("emotion_model.h5")

# Loss vs Val Loss
loss = emotion_model_info.history['loss']
val_loss = emotion_model_info.history['val_loss']
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss', 'val_loss'])
plt.show()

#Accuracy vs Val accuracy
loss = emotion_model_info.history['accuracy']
val_loss = emotion_model_info.history['val_accuracy']
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['acc', 'val_acc'])
plt.show()


# let's create a function that will plot our image
def plot_image(test_image):
  img = mpimg.imread(test_image)
  imgplot = plt.imshow(img)
  plt.show()

#and a function that will outpu the predicted emotion
def predict_emotion(test_image):
  emotion=['Angry','Disgust','Happy','Sad','Surprise']
  test_image = image.load_img(test_image, target_size = (48,48),color_mode='grayscale')
  test_image = image.img_to_array(test_image)
  test_image = np.expand_dims(test_image, axis = 0)
  neural_net_output=emotion_model.predict(test_image)[0]
  print(neural_net_output)
  neural_net_output=neural_net_output.tolist()
  print(emotion [neural_net_output.index(max(neural_net_output))])