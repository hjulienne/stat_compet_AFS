"""
Computation and viz to better understand the dataset
"""
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from sklearn import linear_model
import matplotlib.pyplot as plt 
import numpy.random as rd
import numpy as np
import os.path 
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import normalize
fi = "./data/training_other.csv"
data_soil = pd.read_csv(fi, sep= '\t')

fi_spec = "./data/training_spectra.csv"
spec = pd.read_csv(fi_spec, sep='\t')
ids_feat = range(1,16) + [22] 
ids_targ = range(17,22)

draw = False
#######draw features with the spectrum sum#########:
if draw:
    samp= rd.random_integers(0, data_soil.shape[0], 500)

    scat_feat = scatter_matrix(data_soil.ix[samp, ids_feat], alpha= 0.3, figsize=(15,15), diagonal='kde')
    plt.savefig("./plots/scatter_feat.pdf")

    scat_target =  scatter_matrix(data_soil.ix[samp, ids_targ], alpha= 0.3, figsize=(6,6), diagonal='kde')
    plt.savefig("./plots/scatter_target.pdf")

    #    scat_all =   scatter_matrix(data_soil.ix[samp, 1:22], alpha= 0.3, figsize=(6,6), diagonal='hist')
    #   plt.savefig("./plots/scatter_all.pdf")
    # unreadable...
#######################################################
#represent distribution of every features and targets (with kde)
kde_plots = PdfPages("./plots/KDE_plot.pdf")

for i in data_soil:
    if i != "Depth":
        fig_i = plt.figure()
        data_soil[i].plot(kind="kde")
        plt.xlabel(i)
        plt.savefig(kde_plots, format='pdf')
        plt.clf()
kde_plots.close()


## Apply PCA on feature and on targets separately
## Goal : understand the underlying structure of the data
pca_plots = PdfPages("./plots/PCA_res.pdf")
PCA_feat  = PCA(copy=True)
PCA_targ  = PCA(copy=True)

PCA_feat.fit(data_soil.ix[:,ids_feat] )
#PCA_targ.fit(data_soil.ix[:,ids_targ] )

print(PCA_feat.explained_variance_ratio_)
#print(PCA_targ.explained_variance_ratio_)
plt.figure()
plt.subplot(121)
plt.plot(range(1,17), PCA_feat.explained_variance_ratio_, marker='*')
plt.subplot(122)
plt.plot(range(1,17), PCA_feat.explained_variance_ratio_.cumsum(), marker='*')
plt.savefig(pca_plots, format='pdf')



Xt = PCA_feat.transform(data_soil.ix[:,ids_feat])
samp= rd.random_integers(0, data_soil.shape[0], 500)
for i in range(5):
    plt.figure()
    plt.scatter(Xt[:,i],Xt[:,(i+1)], alpha=0.8, c= data_soil["pH"], cmap=plt.get_cmap("RdYlGn"))
    plt.savefig(pca_plots, format='pdf')

#plt.subplot(121)
#plt.plot(range(1,6), PCA_targ.explained_variance_ratio_, marker="*")
#plt.subplot(122)
#plt.plot(range(1,6), PCA_targ.explained_variance_ratio_.cumsum(), marker="*")
#plt.savefig(pca_plots, format='pdf')
pca_plots.close()

##############Apply PCA on spectra######
#### We apply PCA on sample and not on features (meaning we search the representative spectrum)
pcaS_plt = PdfPages("./plots/PCA_Spec.pdf")
PCA_spec  = PCA(copy=True)


PCA_spec.fit(spec.transpose().values[1:,:])
n_comp = 5
plt.figure()
plt.subplot(121)
plt.plot(range(1, (n_comp+1)), PCA_spec.explained_variance_ratio_[:n_comp], marker='*')
plt.subplot(122)
plt.plot(range(1,(n_comp+1)), PCA_spec.explained_variance_ratio_.cumsum()[:n_comp], marker='*')
plt.savefig(pcaS_plt, format='pdf')

spec_t = PCA_spec.transform(spec.transpose().values[1:,:])
wavelength = spec.columns.values[1:].astype("float")
plt.figure()
cols = np.linspace(0.1,0.9,10)
for j in range(10):
    labe = str(j)
    plt.plot(wavelength, spec_t[:,j], label=j,color=(1,1-cols[j],cols[j]))
    

plt.legend()
plt.savefig(pcaS_plt, format='pdf')
pcaS_plt.close()

####Projection on the 6 first spectra component:
spec_comp = spec.iloc[:,1:].values.dot(spec_t[:,:6])
spec_comp = normalize(spec_comp, axis=1)
spec_comp = pd.DataFrame(spec_comp)
spec_comp.to_csv("./data/spec_component.csv",sep="\t", index=False)
