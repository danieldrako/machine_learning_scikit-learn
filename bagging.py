import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    dt_heart = pd.read_csv('data/heart.csv' )
    #print(dt_heart['target'].describe())

    # #min 0, max 1
    X = dt_heart.drop(['target'], axis=1)
    y = dt_heart['target']

    # #tamano de conjunto de prueba de 35%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
