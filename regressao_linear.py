import numpy as np
from sklearn.linear_model import LinearRegression

def main():
    X = np.array([19, 20, 22, 25, 26, 28, 29, 30, 39, 40, 42, 55, 60]).reshape(-1,1)
    y = np.array([900, 920, 950, 932, 935, 950, 1000, 1250, 2500, 2750, 2800, 3200, 4000])
    reg = LinearRegression().fit(X, y)
    reg.score(X, y)
    print(reg.coef_)
    print(reg.intercept_)
    print(reg.predict(np.array([19, 20, 22]).reshape(-1,1) ))

if __name__ == '__main__':
    main()