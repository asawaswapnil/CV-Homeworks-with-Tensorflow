import PIL
import tensorflow as tf
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix as cfm
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.preprocessing.image import img_to_array
import pickle as pkl


os.environ["CUDA_VISIBLE_DEVICES"]="1"
train_dir = '/opt/cs2770/data/ims/cs2770_images/train_data'
validation_dir = '/opt/cs2770/data/ims/cs2770_images/validation_data'
test_dir = '/opt/cs2770/data/ims/cs2770_images/test_data'

model = VGG16(include_top = True, weights='imagenet')
input_shape = model.layers[0].output_shape[1:3]
print("input shape",input_shape)

def parse(datadir):
	img_list=[]		
	ID_list=[]
	imgs=[]
	i=0
	for root, dirs,fnames in os.walk(datadir):
		for fname in fnames:
			ffname=os.path.join(root,fname) 
			img_list.append(ffname)
			ID_list.append(root.split("/")[-1])
			img=PIL.Image.open( ffname )
			img=img.resize(input_shape,PIL.Image.LANCZOS)
			img=img_to_array(img)
			imgs.append(img)
	imgs=np.array(imgs)
	ID_list=np.array(ID_list)
	print(ID_list)
	unique=sorted(list(set(ID_list)))
	for (index,replacement) in zip(unique,range(len(unique))):
		ID_list[ID_list==index]=replacement
	return imgs, ID_list
test_image_array, test_ID_list=parse(test_dir)
print("test array created")

# dataloaders
datagen_train = ImageDataGenerator(rescale=1./255)
datagen_validation = ImageDataGenerator(rescale=1./255)
datagen_test = ImageDataGenerator(rescale=1./255)

batch_size=8
generator_train = datagen_train.flow_from_directory(directory=train_dir, target_size=input_shape, batch_size = batch_size)
generator_validation = datagen_validation.flow_from_directory(directory=validation_dir, target_size=input_shape, batch_size = batch_size)
generator_test= datagen_test.flow_from_directory(directory=test_dir, target_size=input_shape, batch_size = batch_size,shuffle=False)
print("generators created")

#get the layers of the model and add the rest
transfer_layer = model.get_layer('fc2')
transfered_model = Model(inputs=model.input, outputs=transfer_layer.output)
new_model = Sequential()
new_model.add(transfered_model)
num_classes = 20
new_model.add(Dense(num_classes, activation='softmax'))

#only allow last layer to train
for layer in transfered_model.layers:
	layer.trainable = False

#model2
optimizer = Adam(lr = 1e-5)
loss = 'categorical_crossentropy'
metrics = ['categorical_accuracy']
new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
weight_path = 'best_weight.h5'
checkpoint = ModelCheckpoint(weight_path, monitor='val_categorical_accuracy', verbose=0, save_best_only = True, mode='max', period=2)
tensorboard = TensorBoard(log_dir='./logs', batch_size=batch_size)
callbacks_list = [checkpoint, tensorboard]

epochs = 25
steps_per_epoch = 100
number_of_validation_batches = generator_validation.n / batch_size
history = new_model.fit_generator(generator=generator_train, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=generator_validation, validation_steps=number_of_validation_batches, callbacks=callbacks_list)
acc = history.history['categorical_accuracy']

#create dataloader fr this one. 
prediction = new_model.predict_generator(generator_test)
print(type(prediction),type(test_ID_list))
pred_name = np.argmax(prediction, axis=1)
mat=cfm(generator_test.classes, pred_name)
print("confusion matrix")
print(mat)
true=0.0
for i in range(len(mat)):
	true+=mat[i][i]
acc=true/len(test_ID_list)
print("test accuracy", acc)

