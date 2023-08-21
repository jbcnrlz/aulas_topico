from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd, numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def iris():
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    tree.plot_tree(clf)
    print("oi")

def main():
    cev = np.asarray(pd.read_csv('will_wait.csv'))
    le = preprocessing.LabelEncoder()
    XVs = cev[:,:-1]
    XVs = np.array([le.fit_transform(XVs[:,i]) for i in range(XVs.shape[1])]).T
    YVs = cev[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(XVs,le.fit_transform(YVs).reshape(-1,1))
    #le.fit(YVs)
    clf = DecisionTreeClassifier().fit(x_train,y_train)
    print(classification_report(y_test,clf.predict(x_test)))
    plot_tree(clf,filled=True)
    plt.title("Decision tree trained on all the iris features")
    plt.show()
    print('oi')


if __name__ == '__main__':
    main()