import pandas as pd, numpy as np
from functions import *

def generateTree(csvCols):
    if csvCols is None:
        return None
    dataset = np.asarray(csvCols)
    eB = entropyBoolean( len(dataset[dataset[:,-1] == 'Yes']) / len (dataset) )
    if eB == 0:
        return dataset[:,-1][0]
    classes = set(dataset[:,-1])
    bigger = -500
    idxBigger = -1
    for idx, cName in enumerate(csvCols.columns):
        if cName == 'WillWait':
            break
        distValues = set(dataset[:,idx])
        valsCount = []
        for d in distValues:
            ford = []
            for c in classes:
                ford.append( len(dataset[dataset[:,-1]==c][dataset[dataset[:,-1]==c][:,idx]==d]) )

            valsCount.append(ford)

        ig = informationGain(eB,len(dataset),valsCount)
        if ig >= bigger:
            bigger = ig
            idxBigger = idx
        print("Information gain %s = %f" % (cName,informationGain(eB,len(dataset),valsCount)))
    print('--------------------------------------')
    currNode = {csvCols.columns[idxBigger] : {}}
    vals = set(dataset[:,idxBigger])
    for v in vals:
        nDataset = csvCols[csvCols[csvCols.columns[idxBigger]] == v]
        nDataset = nDataset.drop(csvCols.columns[idxBigger],axis=1)
        currNode[csvCols.columns[idxBigger]][v] = generateTree(nDataset)
    return currNode

def main():
    csvCols = pd.read_csv('datasets\will_wait.csv')
    a = generateTree(csvCols)
    print(a)


if __name__ == '__main__':
    main()