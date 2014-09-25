from features_lib import *
import sys
import pandas as pd
from sklearn.externals import joblib

model_name = sys.argv[1]

fi = "./data/sorted_test_other.csv"
data_soil_test = pd.read_csv(fi, sep= '\t')

fi_spec = "./data/sorted_test_spectra.csv"
spec_test = pd.read_csv(fi_spec, sep='\t')
data_soil_test['spec_amp'] = spec_amp(spec_test)
data_soil_test['Depth'] = data_soil_test['Depth'].map({'Topsoil' : 0, 'Subsoil':1}).astype('int64')
md = joblib.load(model_name)

fo_res = "./data/sample_submission.csv"
fo = "./data/sample_submission_1.csv"
samp_sub =  pd.read_csv(fo_res)
samp_sub.iloc[:,1:] = md.predict(data_soil_test.iloc[:,1:])

samp_sub.to_csv(fo, index=False)
