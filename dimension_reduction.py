import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


class DimensionReduction:
    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        features_name: list = None,
    ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.features_name = features_name
        if self.features_name == None:
            self.features_name = np.arange(1, self.x_train.shape[1] + 1)

        self.x_train_transformed = None
        self.x_test_transformed = None

    def PCA(self, n_components=None) -> tuple[np.ndarray, np.ndarray]:
        self.model = PCA(n_components=n_components)
        x_train_transformed = self.model.fit_transform(self.x_train)
        x_test_transformed = self.model.transform(self.x_test)

        return x_train_transformed, x_test_transformed

    def LDA(self, n_components=None) -> tuple[np.ndarray, np.ndarray]:
        self.model = LDA(n_components=n_components)
        x_train_transformed = self.model.fit_transform(self.x_train, self.y_train)
        x_test_transformed = self.model.transform(self.x_test)

        return x_train_transformed, x_test_transformed

    def plot_explained_variance_ratio(self):
        """
        explained_variance_ratio returns the percentage of variance explained by each component.
        Calculate the cumulative sum to find the explained_varianceratio of the reduced dataset.
        The higher the value the lower is the information loss.

        The first element shows the percentage variance explained
        if the 'n_components'=1, 2nd element shows 'explained_varianceratio' when 'n_components'=2.
        The third element is the 'explained_varianceratio' of the 'n_components' we selected i.e. 3.
        """
        evr = np.cumsum(self.model.explained_variance_ratio_)
        print(evr)

        plt.plot(range(1, len(evr) + 1), evr)
        plt.xticks(range(1, len(evr) + 1))
        plt.title("Explained variance ratio")
        plt.ylabel("Explained variance ratio")
        plt.xlabel("n_components")
        plt.show()

    def plot_ridge_feature_importance(self):
        ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(
            self.x_train, self.y_train
        )
        importance = np.abs(ridge.coef_)

        figure(figsize=(14, 8))
        plt.bar(height=importance, x=self.features_name)
        plt.title("Feature importances via coefficients")
        plt.xticks(rotation=90)
        plt.show()

    def forward_selection(self, n_features_to_select="auto") -> list:
        ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(
            self.x_train, self.y_train
        )

        self.model = SequentialFeatureSelector(
            ridge, n_features_to_select=n_features_to_select, direction="forward"
        ).fit(self.x_train, self.y_train)

        return self.features_name[self.model.get_support()]

    def backward_elimination(self, n_features_to_select="auto") -> list:
        ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(
            self.x_train, self.y_train
        )

        self.model = SequentialFeatureSelector(
            ridge, n_features_to_select=n_features_to_select, direction="backward"
        ).fit(self.x_train, self.y_train)

        return self.features_name[self.model.get_support()]


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    X = df.drop(["0"], axis=1)
    y = df["0"]
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=31
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(np.array(X_train, dtype=float))
    X_test = scaler.transform(np.array(X_test, dtype=float))

    dr = DimensionReduction(
        x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test
    )

    ################### PCA ###################

    # x_train_transformed, x_test_transformed = dr.PCA(n_components=30)
    # print(x_train_transformed.shape)
    # print(x_test_transformed.shape)

    ################### LDA ###################

    # x_train_transformed, x_test_transformed = dr.LDA(n_components=5)
    # print(x_train_transformed.shape)
    # print(x_test_transformed.shape)

    ################### Forward Selection ###################

    # features = dr.forward_selection(n_features_to_select=5)
    # print(features)

    ################### Backward Elimination ###################

    # features = dr.backward_elimination(n_features_to_select=50)
    # print(features)
