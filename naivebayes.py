import pandas as pd, numpy as np

def main():
    datasetPlayTenis = pd.read_csv('datasets/playTennis.csv')
    diaConsultar = ['rain','hot','high','weak']
    trainData = np.array(datasetPlayTenis)[:,1:-1]
    trainLabel = np.array(datasetPlayTenis)[:,-1]
    resultProbs = {}
    total = 0
    for labelValue in np.unique(trainLabel):
        resultProbs[labelValue] = 1
        for idx, d in enumerate(diaConsultar):
            resultProbs[labelValue] *= len(trainData[trainLabel == labelValue][trainData[trainLabel == labelValue][:,idx] == d]) / len(trainData[trainLabel == labelValue])
        
        resultProbs[labelValue] *= len(trainData[trainLabel == labelValue]) / len(trainData)
        total += resultProbs[labelValue]

    for r in resultProbs:
        resultProbs[r] = resultProbs[r] / total
    print(resultProbs)

if __name__ == '__main__':
    main()