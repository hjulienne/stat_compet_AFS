# fit a SVR with brf kernels
#
import pandas as pd
import numpy as np
import numpy.random as rd

import sklearn
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.cross_validation import cross_val_score
from viz_diag import pred_vs_true

#import training set
fi = "./data/training_other.csv"
data_soil = pd.read_csv(fi, sep= '\t')
Nr = data_soil.shape[0]
fi = "./data/spec_component.csv"
data_spec = pd.read_csv(fi, sep= '\t')
data_soil = pd.concat((data_spec, data_soil), axis=1)

#import test set
data_soil_test = pd.read_csv('./data/sorted_test_other.csv', sep='\t')
spec_test = pd.read_csv('./data/spec_component_test.csv',sep="\t")
data_soil_test = pd.concat(( spec_test,data_soil_test), axis=1)

ids_feat =  range(0,6) + range(7,23) #+ [22]# +
ids_targ = range(23,28)
SVR_D={} #Dictionary that contain all SVR model (one per target)

N = data_soil.shape[0]
perm_ids = np.random.permutation(N)
X = data_soil.ix[perm_ids, ids_feat].values # features
Xt = data_soil_test.iloc[:,ids_feat].values
Y = data_soil.ix[perm_ids, ids_targ].values #Target
targ_name = data_soil.ix[:, ids_targ].columns

eps=0.1
ker = 'rbf'

fo_res = "./data/sample_submission.csv"
fo = "./data/sample_submission_3.csv"
samp_sub =  pd.read_csv(fo_res)

#for ker in ['poly', 'rbf']:
#for eps in np.linspace(0.09,0.11,3):
print "cv error for eps %s and %s kernel"%(eps,ker) 
MRC=0
for Ch in [100]:
    print "cv error for C %d eps %s and %s kernel"%(Ch,eps,ker) 
    for i in range(5):
        print targ_name[i]
        mod_name = 'model_'+targ_name[i]
        SVR_D[mod_name] = SVR(kernel=ker, epsilon=eps, C = Ch)
        
        mean_cv = cross_val_score(SVR_D[mod_name], X, Y[:,i], cv=4, scoring='mean_squared_error') 
        SVR_D[mod_name].fit(X,Y[:,i])
        MSE_train = sklearn.metrics.mean_squared_error(Y[:,i], SVR_D[mod_name].predict(X))
    
        print "MSE train %s"%MSE_train
        print "MSE_Cross: %s"%mean_cv.mean()
        MRC+=(-mean_cv.mean())**0.5
        # pred_vs_true(SVR_D[mod_name],X, Y[i] )    
    
        samp_sub[targ_name[i]] = SVR_D[mod_name].predict(Xt).astype(float)

#samp_sub['Sand'] = SVR_D['model_Sand'].predict(Xt.values)
#samp_sub['Ca'] = SVR_D['model_Ca'].predict(Xt.values)
#samp_sub['P'] = SVR_D['model_P'].predict(Xt.values)
#samp_sub['SOC'] = SVR_D['model_SOC'].predict(Xt.values)
samp_sub.to_csv(fo, index=False)
