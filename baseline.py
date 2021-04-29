import numpy as np
import random
import cv2
import os
import shutil
from collections import defaultdict
from tqdm import tqdm
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K 

from fl_implementation_utils import *

# model variable
img_height, img_width = 160,160
batch_size = 128
comms_round = 100

#load data files
data_dir = 'exp1_oulu_at_server'
classes = {'Live': 0, 'PA': 1}
if os.path.isdir(data_dir):
	client_names = [f for f in os.listdir(data_dir) if 'client' in f]
	num_total_images = 0
	for client_name in client_names:
		materials = os.listdir(os.path.join(data_dir,client_name))
		for material in materials:
			image_paths = [f for f in os.listdir(os.path.join(data_dir, client_name, material)) if '.jpg' in f]
			num_total_images += len(image_paths)
	print('total training images: ', num_total_images)
else:
	raise Exception('client dirs dont exist')

#create test datasets
test_ds_siwm, len_test_siwm = make_dataset(os.path.join(data_dir,'test_siwm_1'), batch_size, img_height, img_width)
test_ds_oulu, len_test_oulu = make_dataset(os.path.join(data_dir,'test_oulu_1'), batch_size, img_height, img_width)

	
#create optimizer
lr = 0.01 
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']
optimizer = SGD(lr=lr, 
				decay=lr / comms_round, 
				momentum=0.9
			   ) 


#initialize global model
scnn_global = SimpleCNN()
global_model = scnn_global.build((img_height, img_width), len(classes))
global_model.compile(loss=loss, 
			  optimizer=optimizer, 
			  metrics=metrics)
		

#model checkpoint for saving
checkpoint_filepath = 'baseline_model'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath=checkpoint_filepath,
	save_weights_only=False,
	save_best_only=True,
	verbose=1,
	monitor='val_accuracy',
	mode='max')

#commence global training loop
train_ds, len_train_data, val_ds, len_val_data = make_dataset_baseline(data_dir, batch_size, img_height, img_width)

for epoch in range(comms_round):
	print('\nepoch: ', epoch)

	STEPS_PER_EPOCH = len_train_data // batch_size
	global_model.fit(train_ds, epochs=1, verbose=1, steps_per_epoch=STEPS_PER_EPOCH, callbacks=[model_checkpoint_callback], \
		validation_data=val_ds, validation_steps=len_val_data//batch_size)
	
	loss_siwm, acc_siwm = global_model.evaluate(test_ds_siwm, batch_size=batch_size, steps=len_test_siwm//batch_size)
	print('siwm: epoch: {} | global_acc: {:.3%} | global_loss: {}'.format(epoch, acc_siwm, loss_siwm))

	loss_oulu, acc_oulu = global_model.evaluate(test_ds_oulu, batch_size=batch_size, steps=len_test_oulu//batch_size)
	print('oulu: epoch: {} | global_acc: {:.3%} | global_loss: {}'.format(epoch, acc_oulu, loss_oulu))
	