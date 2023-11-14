import pandas as pd, numpy as np, random
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

def main():
    #dataset = np.array(pd.read_csv('datasets/agiota.csv'))
    #k = 2

    dataset = np.array(pd.read_csv('datasets\Iris.csv'))[:,:-1]
    k = 3

    cWeight = np.zeros((dataset.shape[1],dataset.shape[1]))
    vals = np.array([random.randint(0,int(d.max())) for d in dataset.T])
    np.fill_diagonal(cWeight,vals)
    centroides = np.random.rand(k,dataset.shape[1]).dot(cWeight)    
    centroidsColor = ['tab:cyan','tab:red','tab:olive']
    plt.ion()
    for cVal in range(k):
        plt.scatter(centroides[int(cVal)][0],centroides[int(cVal)][1],c=centroidsColor[int(cVal)])        
        plt.pause(0.05)


    clusters = np.zeros(len(dataset))
    conv = False
    color = ['tab:blue','tab:orange','tab:green']
    while not conv:        
        plt.clf()
        distances = np.zeros((dataset.shape[0],k))
        for idxD, d in enumerate(dataset):
            for idxC, c in enumerate(centroides):
                distances[idxD,idxC] = euclidean(d,c)

            clusters[idxD] = np.argsort(distances[idxD])[0]        
            plt.scatter(dataset[idxD,0],dataset[idxD,1],c=color[int(clusters[idxD])])
        
        for cVal in range(k):
            if len(dataset[clusters == cVal]) > 0:                
                centroides[int(cVal)] = np.mean(dataset[clusters == cVal],axis=0)
            plt.scatter(centroides[int(cVal)][0],centroides[int(cVal)][1],c=centroidsColor[int(cVal)])                
        plt.pause(0.5)
        plt.show()        
        print(distances)


if __name__ == '__main__':
    main()