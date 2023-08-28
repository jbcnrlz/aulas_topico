from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd, numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import graphviz
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def iris():
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(X, y)
    plot_tree(clf,filled=True)
    plt.title("Decision tree trained on all the iris features")
    plt.show()
    print("oi")

def main():
    pdFileRead = pd.read_csv('datasets\will_wait.csv')
    cev = np.asarray(pdFileRead)
    le = preprocessing.LabelEncoder()
    XVs = cev[:,:-1]
    XVs = np.array([le.fit_transform(XVs[:,i]) for i in range(XVs.shape[1])]).T
    YVs = cev[:,-1]
    clf = DecisionTreeClassifier(criterion='entropy')
    scores = cross_val_score(clf,XVs, YVs, cv=2)
    print(scores)
    print("%0.2f acuracia com um desvio padrão de %0.2f" % (scores.mean(), scores.std()))



    
    #x_train, x_test, y_train, y_test = train_test_split(XVs,le.fit_transform(YVs).reshape(-1,1))
    #clf = DecisionTreeClassifier(criterion='entropy').fit(x_train,y_train)
    #print(classification_report(y_test,clf.predict(x_test)))
    '''
    clf = DecisionTreeClassifier(criterion='entropy').fit(XVs,YVs)    
    dot_data = tree.export_graphviz(
        clf,out_file=None,feature_names=list(pdFileRead.columns)[:-1],class_names=['yes','no'],
        filled=True, rounded=True, special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render('teste')
    print('io')
    '''
    #plot_tree(clf,filled=True)
    #plt.title("Decision tree trained on all the iris features")
    #plt.show()
    


if __name__ == '__main__':
    main()