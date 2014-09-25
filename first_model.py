# kind of silly model where we sum up spectra by integrating them
# We try to see if the amplitude of spectrum are informative 
from features_lib import spec_amp
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.externals import joblib
fi = "./data/training_other.csv"
data_soil = pd.read_csv(fi, sep= '\t')

fi_spec = "./data/training_spectra.csv"
spec = pd.read_csv(fi_spec, sep='\t')

data_soil['spec_amp'] = spec_amp(spec)
  
Nr = data_soil.shape[0]

#######draw features with the spectrum sum#########:
#ids = range(1,16) + [22] 
#scat_feat = scatter_matrix(data_soil.ix[samp, ids], alpha= 0.3, figsize=(15,15), diagonal='kde')
#plt.savefig("./plots/scatter_feat.pdf")


#scat_target =  scatter_matrix(data_soil.ix[samp, 17:22], alpha= 0.3, figsize=(6,6), diagonal='kde')
#plt.savefig("./plots/scatter_target.pdf")
#plt.close()

#######################################################
data_soil['Depth'] = data_soil['Depth'].map({'Topsoil' : 0, 'Subsoil':1}).astype('int64')
# We use to ridge regression to weight away correlated features
ids = range(1,17) + [22]
Ns = 800
X = data_soil.ix[:Ns,ids] # features
Y = data_soil.ix[:Ns,17:22] #Target
N = Y.shape[0]

brg = linear_model.Ridge(alpha=1700)#0.005)
brg.fit(X, Y)
MCRMSE_train = sum((((brg.predict(X)-Y)**2).sum()/1157)**0.5)/5

print "MCRMSE_train: %f\n" %MCRMSE_train
#### compute MR on the test set####

X_t = data_soil.ix[Ns:,ids] # features
Y_t = data_soil.ix[Ns:,17:22] #Target
N_t = Y_t.shape[0]
MCRMSE_test = sum((((brg.predict(X_t)-Y_t)**2).sum()/N_t)**0.5)/5
print "MCRMSE_test: %f\n" %MCRMSE_test
joblib.dump(brg, './models/ridge_1.pkl')
