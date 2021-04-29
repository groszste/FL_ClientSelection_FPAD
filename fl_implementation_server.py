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
#(exp1_oulu_at_server, exp1_siwm_at_server)
#(exp2_siwm_replay_at_server, exp2_siwm_print_at_server)
#(exp3_siwm_replay_at_server, exp3_siwm_print_at_server, exp3_oulu_replay_at_server, exp3_oulu_print_at_server)
data_dir = 'exp3_oulu_print_at_server'
classes = {'Live': 0, 'PA': 1}

if not os.path.isdir(data_dir + '/results/server'):
	os.makedirs(data_dir + '/results/server')

if os.path.isdir(data_dir):
	client_names = [f for f in os.listdir(data_dir) if 'server' in f]
	num_total_images = 0
	for client_name in client_names:
		materials = os.listdir(os.path.join(data_dir,client_name))
		for material in materials:
			image_paths = [f for f in os.listdir(os.path.join(data_dir, client_name, material)) if '.jpg' in f]
			num_total_images += len(image_paths)
	print('total training images: ', num_total_images)

if 'exp1' in data_dir:
	#create test datasets for exp1
	test_ds_siwm, len_test_siwm = make_dataset(os.path.join(data_dir,'test_siwm_1'), batch_size, img_height, img_width)
	test_ds_oulu, len_test_oulu = make_dataset(os.path.join(data_dir,'test_oulu_1'), batch_size, img_height, img_width)
elif 'exp2' in data_dir:
	#create test datasets for exp2
	test_ds_siwm_print, len_test_siwm_print = make_dataset(os.path.join(data_dir,'test_siwm_1'), batch_size, img_height, img_width)
	test_ds_siwm_replay, len_test_siwm_replay = make_dataset(os.path.join(data_dir,'test_siwm_2'), batch_size, img_height, img_width)
else:
	#create test datasets for exp3
	test_ds_siwm_print, len_test_siwm_print = make_dataset(os.path.join(data_dir,'test_siwm_1'), batch_size, img_height, img_width)
	test_ds_siwm_replay, len_test_siwm_replay = make_dataset(os.path.join(data_dir,'test_siwm_2'), batch_size, img_height, img_width)
	test_ds_oulu_print, len_test_oulu_print = make_dataset(os.path.join(data_dir,'test_oulu_1'), batch_size, img_height, img_width)
	test_ds_oulu_replay, len_test_oulu_replay = make_dataset(os.path.join(data_dir,'test_oulu_2'), batch_size, img_height, img_width)
	
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

train_ds, len_train_data = make_dataset_baseline_server(data_dir, batch_size, img_height, img_width)
		
# commence global training loop
for epoch in range(comms_round):
	print('\nepoch: ', epoch)

	STEPS_PER_EPOCH = len_train_data // batch_size
	global_model.fit(train_ds, epochs=1, verbose=1, steps_per_epoch=STEPS_PER_EPOCH)

	global_model.save(data_dir + '/results/server/round_' + str(epoch))

	if 'exp1' in data_dir:
		loss_siwm, acc_siwm = global_model.evaluate(test_ds_siwm, batch_size=batch_size, steps=len_test_siwm//batch_size)
		print('siwm: epoch: {} | global_acc: {:.3%} | global_loss: {}'.format(epoch, acc_siwm, loss_siwm))

		loss_oulu, acc_oulu = global_model.evaluate(test_ds_oulu, batch_size=batch_size, steps=len_test_oulu//batch_size)
		print('oulu: epoch: {} | global_acc: {:.3%} | global_loss: {}'.format(epoch, acc_oulu, loss_oulu))

		with open(data_dir + '/results/server/results.txt', 'a+') as f:
			f.write('ROUND ' + str(epoch) + ': SIWM_Acc={:.4} OULU_Acc={:.4} SIWM_Loss={:.4} OULU_Loss={:.4}'.\
				format(acc_siwm, acc_oulu,loss_siwm, loss_oulu) + '\n')
	elif 'exp2' in data_dir:
		STEPS_PER_EPOCH = len_test_siwm_print//batch_size
		loss_siwm_print, acc_siwm_print = global_model.evaluate(test_ds_siwm_print, batch_size=batch_size, steps=STEPS_PER_EPOCH)
		print('siwm print: epoch: {} | global_acc: {:.3%} | global_loss: {}'.format(epoch, acc_siwm_print, loss_siwm_print))

		STEPS_PER_EPOCH = len_test_siwm_replay//batch_size
		loss_siwm_replay, acc_siwm_replay = global_model.evaluate(test_ds_siwm_replay, batch_size=batch_size, steps=STEPS_PER_EPOCH)
		print('siwm replay: epoch: {} | global_acc: {:.3%} | global_loss: {}'.format(epoch, acc_siwm_replay, loss_siwm_replay))

		with open(data_dir + '/results/server/results.txt', 'a+') as f:
			f.write('ROUND ' + str(epoch) + ': SIWM_Print_Acc={:.4} SIWM_Replay_Acc={:.4} SIWM_Print_Loss={:.4} SIWM_Replay_Loss={:.4}\
				 '.format(acc_siwm_print, acc_siwm_replay, loss_siwm_print, loss_siwm_replay) + '\n')
	else:
		STEPS_PER_EPOCH = len_test_siwm_print//batch_size
		loss_siwm_print, acc_siwm_print = global_model.evaluate(test_ds_siwm_print, batch_size=batch_size, steps=STEPS_PER_EPOCH)
		print('siwm print: epoch: {} | global_acc: {:.3%} | global_loss: {}'.format(epoch, acc_siwm_print, loss_siwm_print))

		STEPS_PER_EPOCH = len_test_siwm_replay//batch_size
		loss_siwm_replay, acc_siwm_replay = global_model.evaluate(test_ds_siwm_replay, batch_size=batch_size, steps=STEPS_PER_EPOCH)
		print('siwm replay: epoch: {} | global_acc: {:.3%} | global_loss: {}'.format(epoch, acc_siwm_replay, loss_siwm_replay))
		
		STEPS_PER_EPOCH = len_test_oulu_print//batch_size
		loss_oulu_print, acc_oulu_print = global_model.evaluate(test_ds_oulu_print, batch_size=batch_size, steps=STEPS_PER_EPOCH)
		print('oulu print: epoch: {} | global_acc: {:.3%} | global_loss: {}'.format(epoch, acc_oulu_print, loss_oulu_print))
		
		STEPS_PER_EPOCH = len_test_oulu_replay//batch_size
		loss_oulu_replay, acc_oulu_replay = global_model.evaluate(test_ds_oulu_replay, batch_size=batch_size, steps=STEPS_PER_EPOCH)
		print('oulu replay: epoch: {} | global_acc: {:.3%} | global_loss: {}'.format(epoch, acc_oulu_replay, loss_oulu_replay))

		with open(data_dir + '/results/server/results.txt', 'a+') as f:
			f.write('ROUND ' + str(epoch) + ': SIWM_Print_Acc={:.4} SIWM_Replay_Acc={:.4} OULU_Print_Acc={:.4} OULU_Replay_Acc={:.4} \
				SIWM_Print_Loss={:.4} SIWM_Replay_Loss={:.4} OULU_Print_Loss={:.4} OULU_Replay_Loss={:.4}'.\
				format(acc_siwm_print, acc_siwm_replay, acc_oulu_print, acc_oulu_replay, \
					loss_siwm_print, loss_siwm_replay, loss_oulu_print, loss_oulu_replay) + '\n')


# evaluation
# global_model = tf.keras.models.load_model('exp1_server_model')

# loss_siwm, acc_siwm = global_model.evaluate(test_ds_siwm, batch_size=batch_size, steps=len_test_siwm//batch_size)
# print('siwm: global_acc: {:.3%} | global_loss: {}'.format(acc_siwm, loss_siwm))

# loss_oulu, acc_oulu = global_model.evaluate(test_ds_oulu, batch_size=batch_size, steps=len_test_oulu//batch_size)
# print('oulu: global_acc: {:.3%} | global_loss: {}'.format(acc_oulu, loss_oulu))
	