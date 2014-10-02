# fit a SVR with brf kernels
#
import pandas as pd
import numpy as np
import numpy.random as rd

from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.cross_validation import cross_val_score

fi = "./data/training_other.csv"
data_soil = pd.read_csv(fi, sep= '\t')
Nr = data_soil.shape[0]
#include  sumed up spectrum
fi = "./data/spec_component.csv"
data_spec = pd.read_csv(fi, sep= '\t')
data_soil = pd.concat((data_soil, data_spec), axis=1)
#data_soil = pd.read_csv("./data/training.csv")

# We use to ridge regression to weight away correlated features
ids_feat = range(25,31) + range(1,17) #+ [22]#
ids_targ = range(17,22)
SVR_D={} #Dictionary that contain all SVR model (one per target)

N = data_soil.shape[0]
perm_ids = np.random.permutation(N)
X = data_soil.ix[perm_ids, ids_feat] # features
Y = data_soil.ix[perm_ids, ids_targ] #Target

eps=0.25
ker = 'rbf'

fo_res = "./data/sample_submission.csv"
fo = "./data/sample_submission_2.csv"
samp_sub =  pd.read_csv(fo_res)

#fi = "./data/sorted_test_other.csv"
#data_soil_test = pd.read_csv(fi)#, sep= '\t')

#for ker in ['poly', 'rbf']:
for eps in np.linspace(0.15, 0.25, 7):
    print "cv error for eps %s and %s kernel"%(eps,ker) 
    MRC=0
    for i in Y.columns:
        mod_name = 'model_'+i
        SVR_D[mod_name] = SVR(kernel=ker,epsilon=eps)

        mean_cv = cross_val_score(SVR_D[mod_name] , X.values, Y[i].values, cv=5, scoring='r2')#'mean_squared_error') 
        SVR_D[mod_name].fit(X,Y[i])
  #      samp_sub[i] = SVR_D[mod_name].predict(data_soil_test.iloc[:,1:].values)
    
        print mean_cv.mean()
        MRC+=(-mean_cv.mean())**0.5
#samp_sub.to_csv(fo, index=False)

