import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, KFold

def main():
    csvCols = pd.read_csv('datasets\will_wait.csv')
    data = np.asarray(csvCols)[:,:-1]
    classes = np.asarray(csvCols)[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(data,classes,test_size=0.2,random_state=1)

    kf = KFold(n_splits=5)
    for treino, teste in kf.split(data):
        print("%s %s" % (treino,teste))
        

    print('wololo')
    


if __name__ == '__main__':
    main()