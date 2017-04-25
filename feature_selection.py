import RBM
import tensorflow as tf
import scipy.io as sio
import numpy as np

dataset = sio.loadmat("glioma1.mat")
MA_data = dataset['data']
MA_label = dataset['ylab']

#Normalize for unit variance and zero mean
MA_data = (MA_data - np.mean(MA_data))/np.std(MA_data)

n_items = np.shape(MA_data)[0]
n_feature = np.shape(MA_data)[1]
percentage = 0.5
n_hidden = percentage * n_feature
# rbm = RBM(n_hidden=n_hidden,n_visible=n_feature)
# rbm.build_model()

