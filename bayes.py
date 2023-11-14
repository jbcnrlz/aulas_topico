import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing

def main():
    cev = np.asarray(pd.read_csv('datasets\playTennis.csv'))
    le = preprocessing.LabelEncoder()
    XVs = cev[:,1:-1]
    XVs = np.array([le.fit_transform(XVs[:,i]) for i in range(XVs.shape[1])]).T
    YVs = cev[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(XVs,le.fit_transform(YVs),test_size=0.2)

    gnb = GaussianNB()
    gnbFitted = gnb.fit(x_train, y_train)
    y_pred = gnbFitted.predict(x_test)
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))

if __name__ == '__main__':
    main()