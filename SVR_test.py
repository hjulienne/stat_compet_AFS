#try to reproduce Abhishek solution

import pandas as pd
import numpy as np
import numpy.random as rd
from sklearn import svm, cross_validation, linear_model
from viz_diag import pred_vs_true
import seaborn as sns

train = pd.read_csv('./data/training.csv')
test = pd.read_csv('./data/sorted_test.csv')
targ_name = ['Ca', 'P', 'pH', 'SOC', 'Sand']
labels = train[targ_name].values

samp = pd.read_csv('./data/sample_submission.csv')
train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
test.drop('PIDN', axis=1, inplace=True)

train['Depth'] = train['Depth'].map({'Topsoil' : 0, 'Subsoil':1}).astype('int64')
test['Depth'] = test['Depth'].map({'Topsoil' : 0, 'Subsoil':1}).astype('int64')
xtrain, xtest = np.array(train)[:,:], np.array(test)[:,:]


N = xtrain.shape[0]
perm = np.random.permutation(N)

#def my_kernel(x,y):
 #   return y.

#sup_vec = linear_model.Lasso(alpha=0.1)

for gam in [0.005,0.01, 0.05]:
    sup_vec = svm.SVR(C=100.0, kernel='rbf',gamma=gam)
    MCR=0
    for i in range(5):
        print targ_name[i]
        sup_vec.fit(xtrain, labels[:,i])

        samp[targ_name[i]] = sup_vec.predict(xtest).astype(float)
      #  pred_vs_true(sup_vec, xtrain, labels[:,i])
        mean_cv = cross_validation.cross_val_score(sup_vec, xtrain[perm,:].astype(float), labels[perm,i], cv=4, scoring='mean_squared_error')
        MCR+=(-mean_cv.mean())**0.5

    print "gamma:",gam,"MCR:",(MCR/5)
    

samp.to_csv("./data/samp_4.csv", index=False)
