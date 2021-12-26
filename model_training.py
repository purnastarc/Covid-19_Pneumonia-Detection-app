from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D


main_dir = '/content/datasets/Data'
train_dir =os.path.join(main_dir,'train')
test_dir = os.path.join(main_dir,'test')
train_covid_dir = os.path.join(train_dir,'COVID19')
train_normal_dir = os.path.join(train_dir,'NORMAL')
test_covid_dir = os.path.join(test_dir,'COVID19')
test_normal_dir = os.path.join(test_dir,'NORMAL')


#  data preprocessing and data augmentation, training, testing, validation split
train_datagen = ImageDataGenerator(rescale = 1./255,validation_split = 0.2,zoom_range = 0.2,horizontal_flip = True)                               

validation_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_dir ,target_size = (150,150), subset = 'training', batch_size = 32,class_mode = 'binary')

validation_generator = train_datagen.flow_from_directory(train_dir ,target_size = (150,150),subset = 'validation',batch_size = 32,class_mode = 'binary')

test_generator = test_datagen.flow_from_directory(test_dir ,target_size = (150,150),batch_size = 32,class_mode = 'binary')


# model construction
model = Sequential()

model.add(Conv2D(32,(5,5),padding='SAME',activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(64,(5,5),padding='SAME',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

#compile the model
from tensorflow.keras.optimizers import Adam
model.compile(Adam(lr = 0.001),loss='binary_crossentropy',metrics=['accuracy'])


#training the model
history = model.fit(train_generator,epochs=20,validation_data = validation_generator,validation_steps = 10)


# save the model locally
model.save('model.h5')