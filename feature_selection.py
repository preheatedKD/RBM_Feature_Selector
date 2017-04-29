import csv
import sys
import numpy as np
import scipy.io as sio
import tensorflow as tf

from RBM import RBM
from sklearn.lda import LDA
from scipy.stats import rankdata
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.cross_decomposition import PLSRegression

tf.logging.set_verbosity(tf.logging.ERROR)

#Initial Settings
Epoch = 100
feature2hiddenRatio = 0.5
numFeatures = int(sys.argv[1])
n_components = 3

# dataset = sio.loadmat("glioma1.mat")
# MA_data = dataset['data']
# MA_label = dataset['ylab']
# dataset = sio.loadmat("CLL_SUB_111.mat")
# MA_data = dataset['X']
# MA_label = dataset['Y']
# dataset = sio.loadmat("GLA-BRA-180.mat")
# MA_data = dataset['X']
# MA_label = dataset['Y']
# dataset = sio.loadmat("gcm3.mat")
# MA_data = dataset['data']
# MA_label = dataset['ylab']
dataset = sio.loadmat("MLL2.mat")
MA_data = dataset['data']
MA_label = dataset['ylab']

MA_data, MA_label = shuffle(MA_data, MA_label)

#Normalize for unit variance and zero mean
MA_data = (MA_data - np.mean(MA_data))/np.std(MA_data)

n_feature = np.shape(MA_data)[1]
n_hidden1 = int(feature2hiddenRatio * n_feature)
n_hidden2 = int(feature2hiddenRatio * n_hidden1)
n_hidden3 = int(feature2hiddenRatio * n_hidden2)
# n_hidden4 = int(feature2hiddenRatio * n_hidden3)

with tf.device("/gpu:0"):
	rbm1 = RBM(n_hidden=n_hidden1, n_visible=n_feature, alpha=0.0001, datatype="gaussian")
	rbm2 = RBM(n_hidden=n_hidden2, n_visible=n_hidden1, alpha=0.0001, datatype="binary")
	rbm3 = RBM(n_hidden=n_hidden3, n_visible=n_hidden2, alpha=0.0001, datatype="binary")
	# rbm4 = RBM(n_hidden=n_hidden4, n_visible=n_hidden3, alpha=0.0001, datatype="binary")
	
	for num in range(Epoch):
		new_w, new_hb, new_vb, ReconErr = rbm1.train(MA_data)
		print("Epoch: {}, Reconstruction Error: {}".format(num,ReconErr))

	for num in range(Epoch):
		batch_xs = rbm1.passThrough(MA_data)
		new_w, new_hb, new_vb, ReconErr = rbm2.train(batch_xs)
		print("Epoch: {}, Reconstruction Error: {}".format(num,ReconErr))

	for num in range(Epoch):
		batch_xs = rbm1.passThrough(MA_data)
		batch_xs1 = rbm2.passThrough(batch_xs)
		new_w, new_hb, new_vb, ReconErr = rbm3.train(batch_xs1)
		print("Epoch: {}, Reconstruction Error: {}".format(num,ReconErr))

	temp1 = rbm1.passThrough(MA_data)
	temp2 = rbm2.passThrough(temp1)
	temp3 = rbm3.inference(temp2)
	temp4 = rbm2.passBack(temp3)
	REC = rbm1.passBack(temp4)

	print(np.shape(REC))
	RMSE = np.sqrt(np.mean(np.square(MA_data - REC), axis=0)/float(np.shape(MA_data)[0]))
	print("Shape of RMSE: {}, RMSE {}".format(np.shape(RMSE), RMSE))
	ReconRank = rankdata(RMSE)

	with open('FeatureRank.csv', 'wb') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
		spamwriter.writerow(["Feature Number", "ReconRank", "RMSE"])
		for _ in range(n_feature):
			spamwriter.writerow([_, ReconRank[_], RMSE[_]])

#Choose Top n Features
features = []
temp = []
for data in MA_data:
	for i in range(1,numFeatures+1):
			temp = np.append(temp, data[np.where(ReconRank==i)[0][0]])
	if np.shape(features)[0] == 0:
		features = temp
		temp = []
	else:
		features = np.vstack([features, temp])
		temp = []

#PLS Dimension Reduction
pls2 = PLSRegression(n_components=n_components)
pls2.fit(features, MA_label)
XScore = pls2.transform(features)
# XScore = features

#LDA Classification
kf = KFold(n_splits=5)
kf.get_n_splits(XScore)
mean_acc = 0
for train_index, test_index in kf.split(XScore):
	X_train, X_test = XScore[train_index], XScore[test_index]
	y_train, y_test = MA_label[train_index], MA_label[test_index]
	clf = LDA()
	clf.fit(X_train, y_train)
	Y_predict = clf.predict(X_test)
	for i in range(len(Y_predict)):
		print("Y_Predict {} - Y_Test {}".format(Y_predict[i], y_test[i]))
	acc = accuracy_score(Y_predict, y_test)
	print("Accuracy = {}".format(acc))
	mean_acc = mean_acc + acc

mean_acc = (mean_acc/5) * 100
print("Accuracy is {}".format(mean_acc))

with open("Results/MLL.csv", 'a') as csvFile:
	writer = csv.writer(csvFile)
	writer.writerow([numFeatures, mean_acc])
	csvfile.close()
