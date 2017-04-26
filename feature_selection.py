from RBM import RBM
import tensorflow as tf
import scipy.io as sio
from scipy.stats import rankdata
import numpy as np
import csv
from scipy.stats import rankdata
from sklearn.cross_decomposition import PLSRegression
from sklearn.lda import LDA
from sklearn.model_selection import KFold

Epoch = 100

dataset = sio.loadmat("glioma1.mat")
MA_data = dataset['data']
MA_label = dataset['ylab']

#Normalize for unit variance and zero mean
MA_data = (MA_data - np.mean(MA_data))/np.std(MA_data)

n_items = np.shape(MA_data)[0]
n_feature = np.shape(MA_data)[1]
percentage = 0.5
n_hidden1 = int(percentage * n_feature)
n_hidden2 = int(percentage * n_hidden1)

with tf.device("/gpu:0"):
	rbm1 = RBM(n_hidden=n_hidden1, n_visible=n_feature, alpha=0.0001, datatype="gaussian")
	rbm2 = RBM(n_hidden=n_hidden2, n_visible=n_hidden1, alpha=0.0001, datatype="binary")

	for num in range(Epoch):
		new_w, new_hb, new_vb, ReconErr = rbm1.train(MA_data)
		print("Epoch: {}, Reconstruction Error: {}".format(num,ReconErr))

	for num in range(Epoch):
		batch_xs = rbm1.passThrough(MA_data)
		new_w, new_hb, new_vb, ReconErr = rbm2.train(batch_xs)
		print("Epoch: {}, Reconstruction Error: {}".format(num,ReconErr))

	temp1 = rbm1.passThrough(MA_data)
	temp2 = rbm2.inference(temp1)
	REC = rbm1.passBack(temp2)

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
	for i in range(1,101):
			temp = np.append(temp, data[np.where(ReconRank==i)[0][0]])
	if np.shape(features)[0] == 0:
		features = temp
		temp = []
	else:
		features = np.vstack([features, temp])
		temp = []

#PLS Dimension Reduction
pls2 = PLSRegression(n_components=4)
pls2.fit(features, MA_label)
XScore = pls2.transform(features)

#LDA Classification
kf = KFold(n_splits=5)
kf.get_n_splits(XScore)
for train_index, test_index in kf.split(XScore):
	X_train, X_test = XScore[train_index], XScore[test_index]
	y_train, y_test = MA_label[train_index], MA_label[test_index]
	clf = LDA()
	clf.fit(X_train, y_train)
	Y_predict = clf.predict(X_test)
	acc_iter = np.sum(np.equal(Y_predict, y_test))/np.shape(Y_predict)[0]
	print("Acc_iter = {}".format(acc_iter))