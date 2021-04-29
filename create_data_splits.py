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

create_train_test_split = True
exp1 = True
exp2 = True
exp3 = True

if create_train_test_split:
    create_train_test_split()

if exp1:
    # model variable
    num_siwm_clients = 5
    num_oulu_clients = 6
    img_height, img_width = 160,160
    batch_size = 64
    comms_round = 100
    num_clients_per_round = 5

    #load data files
    data_dir = 'exp1_oulu_at_server'
    # classes = {'Live': 0, 'Paper': 1, 'Replay': 2}
    classes = {'Live': 0, 'PA': 1}

    data_siwm_train, labels_siwm_train, subjIDs_siwm_train = load('exp1_train_subjs_siwm.txt')
    data_siwm_train_lives, labels_siwm_train_lives, subjIDs_siwm_train_lives = load('exp1_train_subjs_siwm_lives.txt')
    data_siwm_test_lives, labels_siwm_test_lives, subjIDs_siwm_test_lives = load('exp1_test_subjs_siwm_lives.txt')
    data_siwm_test, labels_siwm_test, subjIDs_siwm_test = load('exp1_test_subjs_siwm.txt')
    data_oulu_train, labels_oulu_train, subjIDs_oulu_train = load('exp1_train_plus_dev_subjs_oulu.txt')
    data_oulu_test, labels_oulu_test, subjIDs_oulu_test = load('exp1_test_subjs_oulu.txt')


    #create clients
    clients_siwm_pas = create_clients(data_siwm_train, labels_siwm_train, subjIDs_siwm_train, num_clients=num_siwm_clients, initial='client')
    clients_siwm_lives = create_clients(data_siwm_train_lives, labels_siwm_train_lives, subjIDs_siwm_train_lives, num_clients=num_siwm_clients, initial='client')
    clients_siwm = {}
    for clientID in clients_siwm_pas.keys(): 
        clients_siwm[clientID] = clients_siwm_pas[clientID] + clients_siwm_lives[clientID]

    clients_oulu = create_clients(data_oulu_train, labels_oulu_train, subjIDs_oulu_train, num_clients=num_oulu_clients, initial='client')

    clients = {}
    num_clients_counter = 0
    num_total_images = 0
    for client, data in clients_oulu.items():
        clients['client_{}'.format(num_clients_counter)] = data
        num_clients_counter += 1
        num_total_images += len(data)
    for client, data in clients_siwm.items():
        clients['client_{}'.format(num_clients_counter)] = data
        num_clients_counter += 1
        num_total_images += len(data)

    server = {'server': clients.pop('client_0')}  #will be from siwm, switch the ordering above for oulu instead
    num_total_images -= len(server['server'])


    create_client_directories(clients, basedir=data_dir)
    create_client_directories(server, basedir=data_dir)


    # create test datasets
    #siwm
    test_siwm_test_pas = create_clients(data_siwm_test, labels_siwm_test, subjIDs_siwm_test, num_clients=1, initial='test_siwm')
    test_siwm_lives = create_clients(data_siwm_test_lives, labels_siwm_test_lives, subjIDs_siwm_test_lives, num_clients=1, initial='test_siwm')
    test_siwm = {}
    for clientID in test_siwm_test_pas.keys(): 
        test_siwm[clientID] = test_siwm_test_pas[clientID] + test_siwm_lives[clientID]
    create_client_directories(test_siwm, basedir=data_dir)
    #oulu
    test_oulu = create_clients(data_oulu_test, labels_oulu_test, subjIDs_oulu_test, num_clients=1, initial='test_oulu')
    create_client_directories(test_oulu, basedir=data_dir)

    client_names= list(clients.keys())
    print('total training images: ', num_total_images)

if exp2:
    # model variables
    num_siwm_clients_print = 5
    num_siwm_clients_replay = 6

    img_height, img_width = 160,160
    batch_size = 64
    comms_round = 100
    num_clients_per_round = 5

    #load data files
    data_dir = 'exp2_replay_at_server'
    absolute_classes = {'Live': 0, 'Paper': 1, 'Replay': 2}
    classes = {'Live': 0, 'PA': 1}

    data_siwm_train, labels_siwm_train, subjIDs_siwm_train = load('exp1_train_subjs_siwm.txt')
    data_siwm_train_lives, labels_siwm_train_lives, subjIDs_siwm_train_lives = load('exp1_train_subjs_siwm_lives.txt')
    data_siwm_test_lives, labels_siwm_test_lives, subjIDs_siwm_test_lives = load('exp1_test_subjs_siwm_lives.txt')
    data_siwm_test, labels_siwm_test, subjIDs_siwm_test = load('exp1_test_subjs_siwm.txt')
    # data_oulu_train, labels_oulu_train, subjIDs_oulu_train = load('exp1_train_plus_dev_subjs_oulu.txt')
    # data_oulu_test, labels_oulu_test, subjIDs_oulu_test = load('exp1_test_subjs_oulu.txt')


    #create clients
    clients_siwm_prints = create_clients_exp2(data_siwm_train, labels_siwm_train, subjIDs_siwm_train, num_clients=num_siwm_clients_print, initial='client', material=absolute_classes['Paper'])
    clients_siwm_replays = create_clients_exp2(data_siwm_train, labels_siwm_train, subjIDs_siwm_train, num_clients=num_siwm_clients_replay, initial='client', material=absolute_classes['Replay'])
    clients_siwm_lives = create_clients(data_siwm_train_lives, labels_siwm_train_lives, subjIDs_siwm_train_lives, num_clients=num_siwm_clients_print+num_siwm_clients_replay, initial='client')
    clients_siwm = {}

    client_num = 0
    for clientID in clients_siwm_prints.keys(): 
        clients_siwm[client_num] = clients_siwm_prints[clientID] + clients_siwm_lives[clientID]
        client_num += 1
    for clientID in clients_siwm_replays.keys(): 
        clients_siwm[client_num] = clients_siwm_replays[clientID] + clients_siwm_lives[clientID]
        client_num += 1

    # clients_oulu = create_clients(data_oulu_train, labels_oulu_train, subjIDs_oulu_train, num_clients=num_oulu_clients, initial='client')

    clients = {}
    num_clients_counter = 0
    num_total_images = 0
    for client, data in clients_siwm.items():
        clients['client_{}'.format(num_clients_counter)] = data
        num_clients_counter += 1
        num_total_images += len(data)

    server = {'server': clients.pop('client_0')}  #will be from siwm, switch the ordering above for oulu instead
    num_total_images -= len(server['server'])


    create_client_directories(clients, basedir=data_dir)
    create_client_directories(server, basedir=data_dir)


    # create test datasets
    #siwm
    test_siwm_test_prints = create_clients_exp2(data_siwm_test, labels_siwm_test, subjIDs_siwm_test, num_clients=1, initial='test_siwm', material=absolute_classes['Paper'])
    test_siwm_test_replays = create_clients_exp2(data_siwm_test, labels_siwm_test, subjIDs_siwm_test, num_clients=1, initial='test_siwm', material=absolute_classes['Replay'])
    test_siwm_lives = create_clients(data_siwm_test_lives, labels_siwm_test_lives, subjIDs_siwm_test_lives, num_clients=1, initial='test_siwm')
    test_siwm = {}
    client_num = 1
    for clientID in test_siwm_test_prints.keys(): 
        test_siwm['test_siwm_{}'.format(client_num)] = test_siwm_test_prints[clientID] + test_siwm_lives[clientID]
        client_num += 1
    for clientID in test_siwm_test_replays.keys(): 
        test_siwm['test_siwm_{}'.format(client_num)] = test_siwm_test_replays[clientID] + test_siwm_lives[clientID]
        client_num += 1
        
    create_client_directories(test_siwm, basedir=data_dir)

    client_names= list(clients.keys())
    print('total training images: ', num_total_images)