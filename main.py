import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.common import random_state
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# load in data
with open("training.csv") as file:
    df = pd.read_csv(file)

    X = df.iloc[:, 1:-1]
    y = df.iloc[:, 0]

    # splitting data 75:25
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.5, shuffle=False, random_state=0
    )

    for i in [1, 5, 10, 15]:
        # training the data with a KNN 5 neighbours
        cls = KNeighborsClassifier(n_neighbors=i)
        cls.fit(X_train, Y_train)

        y_pred = cls.predict(X_test)

        acc = metrics.accuracy_score(y_pred, Y_test)

        print(acc)

        # display confusion matrix of predictions
        disp = metrics.ConfusionMatrixDisplay.from_predictions(Y_test, y_pred)
        disp.figure_.suptitle(f"Confusion Matrix using {i} Nearest Neighbours")
        name = f"Confusion_Matrix_{i}NN.png"
        disp.figure_.savefig(name)
