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
from tensorflow.python.keras.preprocessing.image import img_to_array
import pickle as pkl

# #initalizations
os.environ["CUDA_VISIBLE_DEVICES"]=""
train_dir = '/opt/cs2770/data/ims/cs2770_images/train_data'
validation_dir = '/opt/cs2770/data/ims/cs2770_images/validation_data'
test_dir = '/opt/cs2770/data/ims/cs2770_images/test_data'
model = VGG16(include_top = True, weights='imagenet')
input_shape = model.layers[0].output_shape[1:3]
print("input shape",input_shape)

#dataloader function, generates numpy array of mimages and it's labels
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

	unique=np.sort(np.array(list(set(ID_list))))
	for (index,replacement) in zip(unique,range(len(unique))):
		ID_list[ID_list==index]=replacement
	return imgs, ID_list


# #dataArrays
test_image_array, test_ID_list=parse(test_dir)
train_image_array, train_ID_list=parse(train_dir)
print("arrays created")


# #get output from VGG
layer_name = 'fc2'
fc2_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)

#get train and test features
feature = fc2_layer_model.predict(train_image_array)
testfeature= fc2_layer_model.predict(test_image_array)
print("features created")


#SVM fitting
scaler = StandardScaler()
scaler.fit(feature)
feature = scaler.transform(feature)
testfeature = scaler.transform(testfeature)
svc = LinearSVC(random_state=5)
svc.fit(testfeature, test_ID_list)
ycap=svc.predict(testfeature)
print(cfm(test_ID_list, ycap))
print('test score is :',svc.score(testfeature, test_ID_list))
