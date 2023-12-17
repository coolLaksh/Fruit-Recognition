import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
import imblearn

seed=321
np.random.seed(seed)
torch.manual_seed(seed)

df = pd.read_csv('./dataset/train.csv')

X_df, Y_df = df.iloc[:,1:-1], df.iloc[:,-1]

from fastai.tabular.all import *
dep_var = 'category'
continuous_vars, categorical_vars = cont_cat_split(X_df, dep_var=dep_var)
waste_cont_features = np.array(continuous_vars)[X_df.loc[:,continuous_vars].nunique() == 1]
waste_cat_features = np.array(categorical_vars)[X_df.loc[:,categorical_vars].nunique() == 1]
# No waste cont features
X_df = X_df.drop(waste_cat_features, axis=1)

X, Y = X_df.to_numpy(), Y_df.to_numpy()


"""
Convert string classes to integer labels
"""
classes = np.unique(Y)
class_to_idx = {classes[i]: i for i in range(len(classes))}
idx_to_class = {class_to_idx[cl]: cl for cl in class_to_idx.keys()}
Y = np.array([class_to_idx[y] for y in Y])


X_train, X_val, Y_train, Y_val= train_test_split(
    X, Y, test_size=0.2, random_state=seed)


"""
Standardising the train data
"""
scaler = StandardScaler()
scaler.fit(X_train)
standard_X_train = scaler.transform(X_train)
standard_X_val = scaler.transform(X_val)

max_performance = 0

for lof_neighbors in range(7,13,1):
    """
    Outlier detection
    """
    clf = LocalOutlierFactor(n_neighbors=lof_neighbors)
    inlier_pred_labels = clf.fit_predict(standard_X_train)
    inlier_X_train = standard_X_train[inlier_pred_labels == 1]
    inlier_Y_train = Y_train[inlier_pred_labels == 1]
    print((inlier_pred_labels == -1).sum())

    for pca_components in range(128, 257, 32):
        """
        Dimensionality reduction
        """
        pca = PCA(n_components=pca_components)
        pca.fit(inlier_X_train)

        pca_X_train = pca.transform(inlier_X_train)
        pca_X_val = pca.transform(standard_X_val)

        for lda_components in range(16, 20, 1):
            lda = LinearDiscriminantAnalysis(n_components=lda_components)
            lda.fit(pca_X_train, inlier_Y_train)

            lda_X_train = lda.transform(pca_X_train)
            lda_X_val = lda.transform(pca_X_val)

            for depth in range(5,26,5):
                for min_samples in range(1,30,10):
                    for n_estimators in range(10,101,30):
                        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=depth, random_state=seed, min_samples_split=min_samples)
                        clf.fit(lda_X_train, inlier_Y_train)
                        train_score = clf.score(lda_X_train, inlier_Y_train)
                        val_score = clf.score(lda_X_val, Y_val)
                        if val_score > max_performance:
                            max_performance = val_score
                            print("Best: ", max_performance, "lof: ", lof_neighbors, "pca_n: ", pca_components, "lda_n: ", lda_components, "at depth:", depth, "min_samples=", min_samples, "n_estimators=", n_estimators)
                        