import matplotlib.pyplot as plt



def pred_vs_true(model,X, Y):
    
    Yp = model.predict(X)
    plt.scatter(Y, Yp)
    mlim = min(Y)
    Mlim = max(Y)
    plt.plot([mlim,Mlim], [mlim,Mlim], 'k-')
    plt.show()
