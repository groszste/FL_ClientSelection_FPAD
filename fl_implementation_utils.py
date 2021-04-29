from glob import glob
import numpy as np
import random
import cv2
import os
import sys
import shutil
from collections import defaultdict
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras import layers

def create_train_test_split():
  #need to split for both oulu and siwm
  material_labels = {'Live': 0, 'Paper': 1, 'Replay': 2}
  material_labels_oulu = {'1': 0, '2': 1, '3': 1, '4': 2, '5': 2}  #1=live, 2,3 = Paper, 4,5=Replay
  
  #siwm
  f_test = 'exp1_test_subjs_siwm.txt'
  f_train = 'exp1_train_subjs_siwm.txt'

  basedir = 'siwm/aligned_160'
  materials_filter = {'Paper', 'Replay'}
  folders = [f for f in os.listdir(basedir) if f in materials_filter]

  subjs_by_material = {}
  for folder in folders:
    subjs = defaultdict(list)
    files = [f for f in os.listdir(os.path.join(basedir,folder)) if '.jpg' in f]
    for file in files:
      material, subj, imageNum = file.split('_')
      subjs[subj].append(os.path.join(basedir,folder,file))
    subjs_by_material[folder] = subjs

  train_subjs_by_material = {}
  test_subjs_by_material = {}
  percent_test = 0.2
  for material, subjs in subjs_by_material.items():
    train_subjs = defaultdict(list)
    test_subjs = defaultdict(list)
    n = len(subjs)
    n_test = int(n * percent_test)
    for i, subj in enumerate(subjs.keys()):
      if i <= n_test:
        test_subjs[subj] = subjs[subj]
      else:
        train_subjs[subj] = subjs[subj]
    train_subjs_by_material[material] = train_subjs
    test_subjs_by_material[material] = test_subjs

  with open(f_test, 'w') as fh:
    for material in test_subjs_by_material.keys():
      for subj, files in test_subjs_by_material[material].items():
        for file in files:
          if 'Paper' in file:
            label = material_labels['Paper']
          elif 'Replay' in file:
            label = material_labels['Replay']
          else:
            label = material_labels['Live']
          fh.write('{} {} {}\n'.format(file, subj, label))
      print('len test_subjs siwm with material {}: {}'.format(material, len(test_subjs_by_material[material].keys())))
  with open(f_train, 'w') as fh:
    for material in train_subjs_by_material.keys():
      for subj, files in train_subjs_by_material[material].items():
        for file in files:
          if 'Paper' in file:
            label = material_labels['Paper']
          elif 'Replay' in file:
            label = material_labels['Replay']
          else:
            label = material_labels['Live']
          fh.write('{} {} {}\n'.format(file, subj, label))
      print('len train_subjs siwm with material {}: {}'.format(material, len(test_subjs_by_material[material].keys())))



  #siwm lives
  percent_test_lives = 0.2
  f_train = 'exp1_train_subjs_siwm_lives.txt'
  basedir = 'siwm/aligned_160'
  materials_filter = {'Train'}
  folders = [f for f in os.listdir(basedir) if f in materials_filter]
  for folder in folders:
    subjs = defaultdict(list)
    files = [f for f in os.listdir(os.path.join(basedir,folder)) if '.jpg' in f]
    for file in files:
      material, subj, imageNum = file.split('_')
      subjs[subj].append('{} {} {}\n'.format(os.path.join(basedir,folder,file), subj, 0))
      
  with open(f_train,'w') as fh:
    for subj, lines in subjs.items():
      random.shuffle(lines)
      # print(int(len(lines)*percent_test))
      # sys.exit()
      lines = random.sample(lines, int(len(lines)*percent_test_lives))
      for line in lines:
        fh.write(line)

  f_test = 'exp1_test_subjs_siwm_lives.txt'
  basedir = 'siwm/aligned_160'
  materials_filter = {'Test'}
  folders = [f for f in os.listdir(basedir) if f in materials_filter]
  subjs_by_material = {}
  for folder in folders:
    subjs = defaultdict(list)
    files = [f for f in os.listdir(os.path.join(basedir,folder)) if '.jpg' in f]
    for file in files:
      material, subj, imageNum = file.split('_')
      subjs[subj].append('{} {} {}\n'.format(os.path.join(basedir,folder,file), subj, 0))
  with open(f_test,'w') as fh:
    for subj, lines in subjs.items():
      random.shuffle(lines)
      # print(int(len(lines)*percent_test))
      lines = random.sample(lines, int(len(lines)*percent_test_lives))
      for line in lines:
        fh.write(line)
  



  #oulu
  f_test = 'exp1_test_subjs_oulu.txt'
  f_train = 'exp1_train_subjs_oulu.txt'
  f_dev = 'exp1_dev_subjs_oulu.txt'

  subjs_test = set()
  subjs_train = set()
  subjs_dev = set()

  basedir = 'oulu/frames/Train_frames'
  with open(f_train,'w') as fh:
    files = [os.path.join(basedir,f) for f in os.listdir(basedir) if '.jpg' in f]
    for file in files:
      cameraID, sessionID, subjID, paType, frame = file.split('/')[-1].split('_')
      # print(cameraID, sessionID, subjID, paType, frame)
      subjs_train.add(subjID)
      label = material_labels_oulu[paType]
      fh.write('{} {} {}\n'.format(file, subjID, label))

  basedir = 'oulu/frames/Test_frames'
  with open(f_test,'w') as fh:
    files = [os.path.join(basedir,f) for f in os.listdir(basedir) if '.jpg' in f]
    for file in files:
      cameraID, sessionID, subjID, paType, frame = file.split('/')[-1].split('_')
      # print(cameraID, sessionID, subjID, paType, frame)
      subjs_test.add(subjID)
      label = material_labels_oulu[paType]
      fh.write('{} {} {}\n'.format(file, subjID, label))

  basedir = 'oulu/frames/Dev_frames'
  with open(f_dev,'w') as fh:
    files = [os.path.join(basedir,f) for f in os.listdir(basedir) if '.jpg' in f]
    for file in files:
      cameraID, sessionID, subjID, paType, frame = file.split('/')[-1].split('_')
      # print(cameraID, sessionID, subjID, paType, frame)
      subjs_dev.add(subjID)
      label = material_labels_oulu[paType]
      fh.write('{} {} {}\n'.format(file, subjID, label))

  assert len(subjs_test.intersection(subjs_dev)) == 0 
  assert len(subjs_train.intersection(subjs_dev)) == 0 
  assert len(subjs_test.intersection(subjs_train)) == 0 

  print('len oulu train subjs: ', len(subjs_train))
  print('len oulu test subjs: ', len(subjs_test))
  print('len oulu dev subjs: ', len(subjs_dev))

def make_dataset(path, batch_size, img_height, img_width):

  def parse_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    return image

  def configure_for_performance(ds):
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

  classes = os.listdir(path)
  filenames = glob(path + '/*/*')
  random.shuffle(filenames)
  labels = [classes.index(name.split('/')[-2]) for name in filenames]

  filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)
  images_ds = filenames_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  labels_ds = tf.data.Dataset.from_tensor_slices(labels)
  ds = tf.data.Dataset.zip((images_ds, labels_ds))
  ds = configure_for_performance(ds)

  return ds, len(filenames)

def make_dataset_baseline(path, batch_size, img_height, img_width):

  def parse_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    return image

  def configure_for_performance(ds):
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

  clients = [f for f in os.listdir(path) if 'test' not in f]
  filenames = []
  for client in clients:
    client_path = os.path.join(path,client)
    classes = os.listdir(client_path)
    filenames += glob(client_path + '/*/*')
  random.shuffle(filenames)
  labels = [classes.index(name.split('/')[-2]) for name in filenames]

  # training dataset
  filenames_ds = tf.data.Dataset.from_tensor_slices(filenames[:int(0.8*len(filenames))])
  images_ds = filenames_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  labels_ds = tf.data.Dataset.from_tensor_slices(labels[:int(0.8*len(labels))])
  ds = tf.data.Dataset.zip((images_ds, labels_ds))
  ds = configure_for_performance(ds)

  # validation dataset
  filenames_ds = tf.data.Dataset.from_tensor_slices(filenames[int(0.8*len(filenames)):])
  images_ds = filenames_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  labels_ds = tf.data.Dataset.from_tensor_slices(labels[int(0.8*len(labels)):])
  ds_val = tf.data.Dataset.zip((images_ds, labels_ds))
  ds_val = configure_for_performance(ds_val)

  return ds, len(filenames[:int(0.8*len(filenames))]), ds_val, len(filenames[int(0.8*len(filenames)):])

def make_dataset_baseline_server(path, batch_size, img_height, img_width):

  def parse_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    return image

  def configure_for_performance(ds):
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

  clients = [f for f in os.listdir(path) if 'server' in f]
  filenames = []
  for client in clients:
    client_path = os.path.join(path,client)
    classes = os.listdir(client_path)
    filenames += glob(client_path + '/*/*')
  random.shuffle(filenames)
  labels = [classes.index(name.split('/')[-2]) for name in filenames]

  # training dataset
  filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)
  images_ds = filenames_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  labels_ds = tf.data.Dataset.from_tensor_slices(labels)
  ds = tf.data.Dataset.zip((images_ds, labels_ds))
  ds = configure_for_performance(ds)

  return ds, len(filenames)

def load(label_file):
    #each line in label_file contains: imgPath, subjID, label
    data = []
    labels = []
    subjIDs = []
    lines = [f.strip() for f in open(label_file).readlines()]
    for line in lines:
        imgPath, subjID, label = line.split(' ')
        data.append(imgPath)
        labels.append(label)
        subjIDs.append(subjID)
    # return a tuple of data, labels, subjIDs
    return data, labels, subjIDs  

def create_clients(image_list, label_list, subj_list, num_clients=10, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as 
                data shards - tuple of images and label lists.
        args: 
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1 
            
    '''

    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]

    #randomize the data
    subjs = np.unique(subj_list)
    # print(subjs)
    random.shuffle(subjs)
    num_subjs_per_client = len(subjs) // num_clients
    subjs_per_client = [subjs[i:i + num_subjs_per_client] for i in range(0, num_subjs_per_client*num_clients, num_subjs_per_client)]

    data = list(zip(image_list, label_list, subj_list))
    random.shuffle(data)

    # print(data)

    # shards = [[]]*num_clients
    client_shards = defaultdict(list)
    # print(shards)
    for imagePath, label, subjID in data:
        for clientID, client_subjs in enumerate(subjs_per_client):
            if subjID in client_subjs:
                client_shards[client_names[clientID]].append((imagePath, label))

    assert(len(client_shards) == len(client_names))
    # return {client_names[i] : shards[i] for i in range(len(client_names))}
    return client_shards

def create_client_directories(clients, basedir='temp/'):
    for clientID, data in clients.items():
        for imgPath, label in data:
            if label != '0':
                # print(label)
                label = '1'
            outdir = os.path.join(basedir,clientID,label)
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
            shutil.copy(imgPath, outdir)
        print(clientID, len(data))

def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    # print(data_shard)
    data, label = zip(*data_shard)
    # print(data)
    # print(label)
    data_list = []
    for data_idx, imgPath in enumerate(data):
        img = cv2.imread(imgPath)
        if img.shape != (160,160,3):
            img = cv2.resize(img, (160,160))
        img = img/255
        data_list.append(img)

    label_list = list(label)
    dataset = tf.data.Dataset.from_tensor_slices((data_list, label_list))
    return dataset.shuffle(len(label)).batch(bs)

class SimpleCNN:
    @staticmethod
    def build(shape, num_classes):
        model = tf.keras.Sequential([
            layers.experimental.preprocessing.Rescaling(1./255, input_shape=(shape[0],shape[1],3)),  #160x160
            layers.Conv2D(64, 3, activation='relu'),  #158x158
            layers.Conv2D(64, 3, activation='relu'),  #156x156
            layers.MaxPooling2D(),  #78x78
            layers.Conv2D(128, 3, activation='relu'),  #76x76
            layers.Conv2D(128, 3, activation='relu'),  #74x74
            layers.MaxPooling2D(), #37x37
            layers.Conv2D(256, 3, activation='relu'),  #35x35
            layers.Conv2D(256, 3, activation='relu'),  #33x33
            layers.Conv2D(256, 3, activation='relu'),  #31x31
            layers.MaxPooling2D(),  #15x15
            layers.Conv2D(512, 3, activation='relu'),  #13x13
            layers.Conv2D(512, 3, activation='relu'),  #11x11
            layers.Conv2D(512, 3, activation='relu'),  #9x9
            layers.MaxPooling2D(),  #4x4
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    

def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final



def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad


def test_model(X_test, Y_test,  model, comm_round):
    cce = keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss

def process_images(image_paths):
    images = []
    for imgPath in image_paths:
        img = cv2.imread(imgPath)
        if img.shape != (160,160,3):
            img = cv2.resize(img, (160,160))
        img = img/255
        images.append(img)
    return images