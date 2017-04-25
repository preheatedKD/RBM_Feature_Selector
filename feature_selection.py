import RBM
import scipy.io as sio
import numpy as np

# sess = tf.Session()
# rbm = RBM(n_hidden=,n_visible=)
# rbm.build_model()

mat = sio.loadmat("glioma1.mat")
MA_data = mat['data']
#Need to normalize
n_items = np.shape(MA_data)[0]
n_feature = np.shape(MA_data)[1]
percentage = 0.5
n_hidden = percentage * n_feature
rbm = RBM(n_hidden=n_hidden,n_visible=n_feature)
rbm.build_model()

