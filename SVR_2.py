#try to reproduce Abhishek solution

import pandas as pd
import numpy as np
import numpy as rd
from sklearn import svm, cross_validation,decomposition
from viz_diag import pred_vs_true

train = pd.read_csv('./data/training.csv')
test = pd.read_csv('./data/sorted_test.csv')
targ_name = ['Ca', 'P', 'pH', 'SOC', 'Sand']
labels = train[targ_name].values

samp = pd.read_csv('./data/sample_submission.csv')
train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
test.drop('PIDN', axis=1, inplace=True)

spec_train, spec_test = np.array(train)[:,:3578], np.array(test)[:,:3578]
#####Reduce the number of features:

###################
# Reduce spectrum to eight component
PCA_train = decomposition.PCA(n_components=8, copy=True)
PCA_train.fit(spec_train.transpose())
repr_spec = PCA_train.transform(spec_train.transpose())
#representative spectrum according to PCA
xtrain= spec_train.dot(repr_spec)
xtest= spec_test.dot(repr_spec)
#PCA_test
###################
N = xtrain.shape[0]
perm = np.random.permutation(N)
sup_vec = svm.SVR(C=100.0)
MCR=0
for i in range(5):
    print targ_name[i]
    sup_vec.fit(xtrain, labels[:,i])

    samp[targ_name[i]] = sup_vec.predict(xtest).astype(float)
    pred_vs_true(sup_vec,xtrain, labels[:,i])
    mean_cv = cross_validation.cross_val_score(sup_vec, xtrain[perm,:], labels[perm,i], cv=4, scoring='mean_squared_error')
    MCR+=(-mean_cv.mean())**0.5

    print mean_cv
    

samp.to_csv("./data/samp_4.csv", index=False)
