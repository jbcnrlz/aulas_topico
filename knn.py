import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing

def main():
    cev = np.asarray(pd.read_csv('adult.csv'))
    le = preprocessing.LabelEncoder()
    XVs = cev[:,:-1]
    XVs = np.array([le.fit_transform(XVs[:,i]) for i in range(XVs.shape[1])]).T
    YVs = cev[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(XVs,le.fit_transform(YVs))

    gnb = KNeighborsClassifier(n_neighbors=3)
    y_pred = gnb.fit(x_train, y_train).predict(x_test)
    print(classification_report(y_test,y_pred))

if __name__ == '__main__':
    main()