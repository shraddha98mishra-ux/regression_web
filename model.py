import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import os

# Dataset
data = pd.DataFrame({
    "Area":[720,500,810,740,940,860,590,830,650,330,880,810],
    "Rooms":[2,1,3,2,3,2,1,3,2,1,2,3],
    "Floor":[15,12,7,10,8,9,6,7,8,5,9,10],
    "Cost":[50,35,60,55,80,70,40,65,52,25,72,62]
})

X = data[["Area","Rooms","Floor"]].values
y = data["Cost"].values

def run_model(method, lam):
    if method == "OLS":
        model = LinearRegression()
    elif method == "Ridge":
        model = Ridge(alpha=lam)
    elif method == "Lasso":
        model = Lasso(alpha=lam)
    elif method == "Elastic":
        model = ElasticNet(alpha=lam, l1_ratio=0.5)

    model.fit(X, y)
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return model.coef_, mse, r2

def generate_plots(method):
    lambdas = np.linspace(0.001, 10, 50)
    w1, w2, w3, mses = [], [], [], []

    for lam in lambdas:
        coef, mse, _ = run_model(method, lam)
        w1.append(coef[0])
        w2.append(coef[1])
        w3.append(coef[2])
        mses.append(mse)

    if not os.path.exists("static"):
        os.mkdir("static")

    plt.figure()
    plt.plot(lambdas, w1, label="Area")
    plt.plot(lambdas, w2, label="Rooms")
    plt.plot(lambdas, w3, label="Floor")
    plt.legend()
    plt.savefig("static/weights.png")
    plt.close()

    plt.figure()
    plt.plot(lambdas, mses)
    plt.savefig("static/mse.png")
    plt.close()
