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
num_siwm_clients = 6
num_oulu_clients = 5
img_height, img_width = 160,160
batch_size = 128
comms_round = 100
num_clients_per_round = 5

# client selection algorithm (options = 'FedAvg', 'CSFedAvg', 'CSFedAvgInverse', 'PowerOfChoice')
# FedAvg: Selects random clients
# CSFedAvg: Selects clients having highest IIDness with server
# CSFedAvgInverse: Selects clients having least IIDness with server
# PowerOfChoice: Selects clients with higher local loss
cs_algorithm = "FedAvg"

#load data files
data_dir = 'exp1_oulu_at_server' #(exp1_oulu_at_server, exp1_siwm_at_server)
classes = {'Live': 0, 'PA': 1}
server_model_path = data_dir + '/results/server/round_0'

if not os.path.isdir(data_dir + '/results/' + cs_algorithm):
	os.makedirs(data_dir + '/results/' + cs_algorithm)

if os.path.isdir(data_dir):
	client_names_all = [f for f in os.listdir(data_dir) if 'client' in f]
	num_total_images = 0
	for client_name in client_names_all:
		materials = os.listdir(os.path.join(data_dir,client_name))
		for material in materials:
			image_paths = [f for f in os.listdir(os.path.join(data_dir, client_name, material)) if '.jpg' in f]
			num_total_images += len(image_paths)
	print('total training images: ', num_total_images)

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

if (cs_algorithm == 'FedAvg'):
	#initialize/load global model
	global_model = tf.keras.models.load_model(server_model_path)

	#commence global training loop
	for comm_round in range(comms_round):
		print('\ncomm round: ', comm_round)
				
		# get the global model's weights - will serve as the initial weights for all local models
		global_weights = global_model.get_weights()
		
		#initial list to collect local model weights after scalling
		scaled_local_weight_list = list()

		client_names = random.sample(client_names_all, num_clients_per_round)

		wd = []
		
		print('clients selected this round: ', client_names)
		for client_name in client_names:
			if int(client_name.split('_')[-1]) <= num_siwm_clients:
				print('client name: {} --- {}'.format(client_name,'siwm'))
			else:
				print('client name: {} --- {}'.format(client_name,'oulu'))
			client_data_dir = os.path.join(data_dir,client_name)
			client_data_paths = [f for f in os.listdir(client_data_dir) if '.jpg' in f]

			scnn_local = SimpleCNN()
			local_model = scnn_local.build((img_height, img_width), len(classes))
			local_model.compile(loss=loss, 
						  optimizer=optimizer, 
						  metrics=metrics)
			
			#set local model weight to the weight of the global model
			local_model.set_weights(global_weights)

			train_ds, len_client_data = make_dataset(client_data_dir, batch_size, img_height, img_width)

			# len_client_data = tf.data.experimental.cardinality(train_ds).numpy()
			STEPS_PER_EPOCH = len_client_data // batch_size
			local_model.fit(train_ds, epochs=1, verbose=1, steps_per_epoch=STEPS_PER_EPOCH)
			
			#scale the model weights and add to list
			scaling_factor = len_client_data / num_total_images
			scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
			scaled_local_weight_list.append(scaled_weights)

			#get weight divergence
			weight_diff = np.array(local_model.get_weights(), dtype=object) - np.array(global_weights, dtype=object)
			num = np.linalg.norm([np.linalg.norm(weight_diff[i]) for i in range(weight_diff.shape[0])])
			den = np.linalg.norm([np.linalg.norm(np.array(global_weights, dtype=object)[i]) for i in range(np.array(global_weights, dtype=object).shape[0])])
			wd_client = num / den
			wd.append(wd_client)
			
			#clear session to free memory after each communication round
			K.clear_session()
			
		#to get the average over all the local model, we simply take the sum of the scaled weights
		average_weights = sum_scaled_weights(scaled_local_weight_list)
		
		#update global model 
		global_model.set_weights(average_weights)

		STEPS_PER_EPOCH = len_test_siwm // batch_size
		loss_siwm, acc_siwm = global_model.evaluate(test_ds_siwm, batch_size=batch_size, steps=STEPS_PER_EPOCH)
		print('siwm: comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc_siwm, loss_siwm))

		STEPS_PER_EPOCH = len_test_oulu // batch_size
		loss_oulu, acc_oulu = global_model.evaluate(test_ds_oulu, batch_size=batch_size, steps=STEPS_PER_EPOCH)
		print('oulu: comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc_oulu, loss_oulu))

		global_model.save(data_dir + '/results/' + cs_algorithm + '/round_' + str(comm_round))

		with open(data_dir + '/results/' + cs_algorithm + '/results.txt', 'a+') as f:
			f.write('ROUND ' + str(comm_round) + ': SIWM_Acc={:.4} OULU_Acc={:.4} SIWM_Loss={:.4} OULU_Loss={:.4} WD={:.4} '.\
				format(acc_siwm, acc_oulu,loss_siwm, loss_oulu, np.mean(wd)) + 'Clients=' + ','.join(client_names) + '\n')

elif (cs_algorithm == 'CSFedAvg'):
	#initialize/load global model
	scnn_global = SimpleCNN()
	global_model = scnn_global.build((img_height, img_width), len(classes))
	global_model.compile(loss=loss, 
				  optimizer=optimizer, 
				  metrics=metrics)

	#get training and validation data for server
	train_ds, len_train_data = make_dataset_baseline_server(data_dir, batch_size, img_height, img_width)
	num_total_images = num_total_images + len_train_data

	#commence global training loop
	for comm_round in range(comms_round):
		print('\ncomm round: ', comm_round)

		#train global model with server data (serves as reference IID data)
		STEPS_PER_EPOCH = len_train_data // batch_size
		global_model.fit(train_ds, epochs=1, verbose=1, steps_per_epoch=STEPS_PER_EPOCH)
				
		# get the global model's weights - will serve as the initial weights for all local models
		global_weights = global_model.get_weights()
		
		#initial list to collect local model weights after scalling
		scaled_local_weight_list = list()

		client_names = client_names_all
		
		# print('clients selected this round: ', client_names)
		clients_weight_divergence = list()
		for client_name in client_names:
			if int(client_name.split('_')[-1]) <= num_siwm_clients:
				print('client name: {} --- {}'.format(client_name,'siwm'))
			else:
				print('client name: {} --- {}'.format(client_name,'oulu'))
			client_data_dir = os.path.join(data_dir,client_name)
			client_data_paths = [f for f in os.listdir(client_data_dir) if '.jpg' in f]

			scnn_local = SimpleCNN()
			local_model = scnn_local.build((img_height, img_width), len(classes))
			local_model.compile(loss=loss, 
						  optimizer=optimizer, 
						  metrics=metrics)
			
			#set local model weight to the weight of the global model
			local_model.set_weights(global_weights)

			train_ds, len_client_data = make_dataset(client_data_dir, batch_size, img_height, img_width)

			# len_client_data = tf.data.experimental.cardinality(train_ds).numpy()
			STEPS_PER_EPOCH = len_client_data // batch_size
			local_model.fit(train_ds, epochs=1, verbose=1, steps_per_epoch=STEPS_PER_EPOCH)
			
			#scale the model weights and add to list
			scaling_factor = len_client_data / num_total_images
			scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
			scaled_local_weight_list.append(scaled_weights)

			#get weight divergence
			weight_diff = np.array(local_model.get_weights(), dtype=object) - np.array(global_weights, dtype=object)
			num = np.linalg.norm([np.linalg.norm(weight_diff[i]) for i in range(weight_diff.shape[0])])
			den = np.linalg.norm([np.linalg.norm(np.array(global_weights, dtype=object)[i]) for i in range(np.array(global_weights, dtype=object).shape[0])])
			wd = num / den
			clients_weight_divergence.append(wd)
			
			#clear session to free memory after each communication round
			K.clear_session()
		
		#select best clients based on weight divergence (lesser is preferred)
		sorted_indices = np.argsort(clients_weight_divergence)
		selected_scaled_local_weight_list = [scaled_local_weight_list[i] for i in sorted_indices[:num_clients_per_round]]

		#to get the average over all the local model, we simply take the sum of the scaled weights
		scaling_factor = (len_train_data) / num_total_images
		global_model_scaled_weight = scale_model_weights(global_weights, scaling_factor)
		selected_scaled_local_weight_list.append(global_model_scaled_weight)
		average_weights = sum_scaled_weights(selected_scaled_local_weight_list)
		
		#update global model 
		global_model.set_weights(average_weights)

		STEPS_PER_EPOCH = len_test_siwm // batch_size
		loss_siwm, acc_siwm = global_model.evaluate(test_ds_siwm, batch_size=batch_size, steps=STEPS_PER_EPOCH)
		print('siwm: comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc_siwm, loss_siwm))

		STEPS_PER_EPOCH = len_test_oulu // batch_size
		loss_oulu, acc_oulu = global_model.evaluate(test_ds_oulu, batch_size=batch_size, steps=STEPS_PER_EPOCH)
		print('oulu: comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc_oulu, loss_oulu))

		global_model.save(data_dir + '/results/' + cs_algorithm + '/round_' + str(comm_round))

		client_names = np.array(client_names)[sorted_indices[:num_clients_per_round]]
		clients_weight_divergence = np.array(clients_weight_divergence)[sorted_indices[:num_clients_per_round]]
		
		with open(data_dir + '/results/' + cs_algorithm + '/results.txt', 'a+') as f:
			f.write('ROUND ' + str(comm_round) + ': SIWM_Acc={:.4} OULU_Acc={:.4} SIWM_Loss={:.4} OULU_Loss={:.4} WD={:.4} '.\
				format(acc_siwm, acc_oulu,loss_siwm, loss_oulu, np.mean(clients_weight_divergence)) + 'Clients=' + ','.join(client_names) + '\n')

elif (cs_algorithm == 'CSFedAvgInverse'):
	#initialize/load global model
	scnn_global = SimpleCNN()
	global_model = scnn_global.build((img_height, img_width), len(classes))
	global_model.compile(loss=loss, 
				  optimizer=optimizer, 
				  metrics=metrics)

	#get training and validation data for server
	train_ds, len_train_data = make_dataset_baseline_server(data_dir, batch_size, img_height, img_width)
	num_total_images = num_total_images + len_train_data

	#commence global training loop
	for comm_round in range(comms_round):
		print('\ncomm round: ', comm_round)

		#train global model with server data (serves as reference IID data)
		STEPS_PER_EPOCH = len_train_data // batch_size
		global_model.fit(train_ds, epochs=1, verbose=1, steps_per_epoch=STEPS_PER_EPOCH, \
			validation_data=val_ds, validation_steps=len_val_data//batch_size)
				
		# get the global model's weights - will serve as the initial weights for all local models
		global_weights = global_model.get_weights()
		
		#initial list to collect local model weights after scalling
		scaled_local_weight_list = list()

		client_names = client_names_all
		
		# print('clients selected this round: ', client_names)
		clients_weight_divergence = list()
		for client_name in client_names:
			if int(client_name.split('_')[-1]) <= num_siwm_clients:
				print('client name: {} --- {}'.format(client_name,'siwm'))
			else:
				print('client name: {} --- {}'.format(client_name,'oulu'))
			client_data_dir = os.path.join(data_dir,client_name)
			client_data_paths = [f for f in os.listdir(client_data_dir) if '.jpg' in f]

			scnn_local = SimpleCNN()
			local_model = scnn_local.build((img_height, img_width), len(classes))
			local_model.compile(loss=loss, 
						  optimizer=optimizer, 
						  metrics=metrics)
			
			#set local model weight to the weight of the global model
			local_model.set_weights(global_weights)

			train_ds, len_client_data = make_dataset(client_data_dir, batch_size, img_height, img_width)

			# len_client_data = tf.data.experimental.cardinality(train_ds).numpy()
			STEPS_PER_EPOCH = len_client_data // batch_size
			local_model.fit(train_ds, epochs=1, verbose=1, steps_per_epoch=STEPS_PER_EPOCH)
			
			#scale the model weights and add to list
			scaling_factor = len_client_data / num_total_images
			scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
			scaled_local_weight_list.append(scaled_weights)

			#get weight divergence
			weight_diff = np.array(local_model.get_weights(), dtype=object) - np.array(global_weights, dtype=object)
			num = np.linalg.norm([np.linalg.norm(weight_diff[i]) for i in range(weight_diff.shape[0])])
			den = np.linalg.norm([np.linalg.norm(np.array(global_weights, dtype=object)[i]) for i in range(np.array(global_weights, dtype=object).shape[0])])
			wd = num / den
			clients_weight_divergence.append(wd)
			
			#clear session to free memory after each communication round
			K.clear_session()
		
		#select best clients based on weight divergence (more is preferred)
		sorted_indices = np.flip(np.argsort(clients_weight_divergence))
		selected_scaled_local_weight_list = [scaled_local_weight_list[i] for i in sorted_indices[:num_clients_per_round]]

		#to get the average over all the local model, we simply take the sum of the scaled weights
		scaling_factor = (len_train_data) / num_total_images
		global_model_scaled_weight = scale_model_weights(global_weights, scaling_factor)
		selected_scaled_local_weight_list.append(global_model_scaled_weight)
		average_weights = sum_scaled_weights(selected_scaled_local_weight_list)
		
		#update global model 
		global_model.set_weights(average_weights)

		STEPS_PER_EPOCH = len_test_siwm // batch_size
		loss_siwm, acc_siwm = global_model.evaluate(test_ds_siwm, batch_size=batch_size, steps=STEPS_PER_EPOCH)
		print('siwm: comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc_siwm, loss_siwm))

		STEPS_PER_EPOCH = len_test_oulu // batch_size
		loss_oulu, acc_oulu = global_model.evaluate(test_ds_oulu, batch_size=batch_size, steps=STEPS_PER_EPOCH)
		print('oulu: comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc_oulu, loss_oulu))

		global_model.save(data_dir + '/results/' + cs_algorithm + '/round_' + str(comm_round))

		client_names = np.array(client_names)[sorted_indices[:num_clients_per_round]]
		clients_weight_divergence = np.array(clients_weight_divergence)[sorted_indices[:num_clients_per_round]]

		with open(data_dir + '/results/' + cs_algorithm + '/results.txt', 'a+') as f:
			f.write('ROUND ' + str(comm_round) + ': SIWM_Acc={:.4} OULU_Acc={:.4} SIWM_Loss={:.4} OULU_Loss={:.4} WD={:.4} '.\
				format(acc_siwm, acc_oulu,loss_siwm, loss_oulu, np.mean(clients_weight_divergence)) + 'Clients=' + ','.join(client_names) + '\n')

elif (cs_algorithm == 'PowerOfChoice'):
	#initialize/load global model
	scnn_global = SimpleCNN()
	global_model = scnn_global.build((img_height, img_width), len(classes))
	global_model.compile(loss=loss, 
				  optimizer=optimizer, 
				  metrics=metrics)

	#commence global training loop
	for comm_round in range(comms_round):
		print('\ncomm round: ', comm_round)

		#train global model with server data (serves as reference IID data)
		global_model = tf.keras.models.load_model(server_model_path)
				
		# get the global model's weights - will serve as the initial weights for all local models
		global_weights = global_model.get_weights()
		
		#initial list to collect local model weights after scalling
		scaled_local_weight_list = list()

		client_names = client_names_all
		
		# print('clients selected this round: ', client_names)
		clients_local_loss = list()
		clients_weight_divergence = list()
		for client_name in client_names:
			if int(client_name.split('_')[-1]) <= num_siwm_clients:
				print('client name: {} --- {}'.format(client_name,'siwm'))
			else:
				print('client name: {} --- {}'.format(client_name,'oulu'))
			client_data_dir = os.path.join(data_dir,client_name)
			client_data_paths = [f for f in os.listdir(client_data_dir) if '.jpg' in f]

			scnn_local = SimpleCNN()
			local_model = scnn_local.build((img_height, img_width), len(classes))
			local_model.compile(loss=loss, 
						  optimizer=optimizer, 
						  metrics=metrics)
			
			#set local model weight to the weight of the global model
			local_model.set_weights(global_weights)

			train_ds, len_client_data = make_dataset(client_data_dir, batch_size, img_height, img_width)

			# len_client_data = tf.data.experimental.cardinality(train_ds).numpy()
			STEPS_PER_EPOCH = len_client_data // batch_size
			history = local_model.fit(train_ds, epochs=1, verbose=1, steps_per_epoch=STEPS_PER_EPOCH)
			
			#scale the model weights and add to list
			scaling_factor = len_client_data / num_total_images
			scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
			scaled_local_weight_list.append(scaled_weights)

			#get client local loss
			clients_local_loss.append(history.history['loss'][0])

			#get weight divergence
			weight_diff = np.array(local_model.get_weights(), dtype=object) - np.array(global_weights, dtype=object)
			num = np.linalg.norm([np.linalg.norm(weight_diff[i]) for i in range(weight_diff.shape[0])])
			den = np.linalg.norm([np.linalg.norm(np.array(global_weights, dtype=object)[i]) for i in range(np.array(global_weights, dtype=object).shape[0])])
			wd = num / den
			clients_weight_divergence.append(wd)
			
			#clear session to free memory after each communication round
			K.clear_session()
		
		#select best clients based on local loss (higher is preferred)
		sorted_indices = np.flip(np.argsort(clients_local_loss))
		selected_scaled_local_weight_list = [scaled_local_weight_list[i] for i in sorted_indices[:num_clients_per_round]]

		#to get the average over all the local model, we simply take the sum of the scaled weights
		average_weights = sum_scaled_weights(selected_scaled_local_weight_list)
		
		#update global model 
		global_model.set_weights(average_weights)

		STEPS_PER_EPOCH = len_test_siwm // batch_size
		loss_siwm, acc_siwm = global_model.evaluate(test_ds_siwm, batch_size=batch_size, steps=STEPS_PER_EPOCH)
		print('siwm: comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc_siwm, loss_siwm))

		STEPS_PER_EPOCH = len_test_oulu // batch_size
		loss_oulu, acc_oulu = global_model.evaluate(test_ds_oulu, batch_size=batch_size, steps=STEPS_PER_EPOCH)
		print('oulu: comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc_oulu, loss_oulu))

		global_model.save(data_dir + '/results/' + cs_algorithm + '/round_' + str(comm_round))

		client_names = np.array(client_names)[sorted_indices[:num_clients_per_round]]
		clients_weight_divergence = np.array(clients_weight_divergence)[sorted_indices[:num_clients_per_round]]

		with open(data_dir + '/results/' + cs_algorithm + '/results.txt', 'a+') as f:
			f.write('ROUND ' + str(comm_round) + ': SIWM_Acc={:.4} OULU_Acc={:.4} SIWM_Loss={:.4} OULU_Loss={:.4} WD={:.4} '.\
				format(acc_siwm, acc_oulu,loss_siwm, loss_oulu, np.mean(clients_weight_divergence)) + 'Clients=' + ','.join(client_names) + '\n')
	