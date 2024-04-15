import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X.drop(inplace=True, columns=self.columns)
        return X

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X[self.columns] = np.log(X[self.columns])
        return X
    

class SquareRootTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X[self.columns] = np.sqrt(X[self.columns])
        return X
    

class CubeRootTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X[self.columns] = np.cbrt(X[self.columns])
        return X

class ZScoreTrimmer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str], z_score_threshold: int = 3) -> None: 
        self.columns = columns
        self.z_score_threshold = z_score_threshold

    def fit(self, X: pd.DataFrame, y=None): 
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        total_outliers = 0
        for column in self.columns:
            z_scores = zscore(X[column], nan_policy="omit")
            outlier_indices = np.where(np.abs(z_scores) > self.z_score_threshold)[0]
            total_outliers += len(outlier_indices)
            X = X.drop(X.index[outlier_indices])
        
        print(f"ZScoreTrimmer - Trimmed a total of {len(self.columns)} columns and removed {total_outliers} outliers.")
        X.reset_index(inplace=True, drop=True)
        return X

class UpperBoundTrimmer(BaseEstimator, TransformerMixin):
    def __init__(self, column_boundaries: dict[str, float]) -> None: 
        self.column_boundaries = column_boundaries

    def fit(self, X: pd.DataFrame, y=None): 
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for column, upperbound in self.column_boundaries.items():
            X_filtered = X.loc[(X[column] <= upperbound) | (pd.isna(X[column]))]
            n_outliers = len(X) - len(X_filtered)
            print(f"UpperBoundTrimmer - Column {column}: {n_outliers} outliers removed. Max value of {X[column].max()} reduced to {X_filtered[column].max()}.")
            X = X_filtered
        
        X.reset_index(inplace=True, drop=True)
        return X

class LowerBoundTrimmer(BaseEstimator, TransformerMixin):
    def __init__(self, column_boundaries: dict[str, float]) -> None: 
        self.column_boundaries = column_boundaries

    def fit(self, X: pd.DataFrame, y=None): 
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for column, lowerbound in self.column_boundaries.items():
            X_filtered = X.loc[(X[column] > lowerbound) | (pd.isna(X[column]))]
            n_outliers = len(X) - len(X_filtered)
            print(f"LowerBoundTrimmer - Column {column}: {n_outliers} outliers removed. Min value of {X[column].min()} reduced to {X_filtered[column].min()}.")
            X = X_filtered
            
        X.reset_index(inplace=True, drop=True)
        return X


class KNNColumnImputer(KNNImputer):
    def __init__(self, n_neighbors:int, columns: list[str]) -> None:
        self.columns = columns
        super().__init__(n_neighbors=n_neighbors)
    
    def fit(self, X: pd.DataFrame, y=None):
        self.imputer = super().fit(X[self.columns], y)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_subset = X[self.columns]
        X_subset = super().transform(X_subset)
        X[self.columns] = X_subset
        return X

class FillNaColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_value:str, columns: list[str]) -> None:
        self.columns = columns
        self.fill_value = fill_value
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for column in self.columns:
            X[column] = X[column].fillna(self.fill_value)

        return X

class OneHotColumnEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        for column in self.columns:
            if column not in X.columns:
                raise ValueError(f"Column '{column}' not found in the DataFrame.")

            column_cat = X[[column]]
            encoded_data = encoder.fit_transform(column_cat)

            new_columns = [f"{column}_{value}" for value in encoder.get_feature_names_out()]
            X[new_columns] = encoded_data

        X = X.drop(columns=self.columns)
        return X
    
class OrdinalColumnEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        encoder = OrdinalEncoder()
        for column in self.columns:
            if column not in X.columns:
                raise ValueError(f"Column '{column}' not found in the DataFrame.")

            column_cat = X[[column]]
            encoded_data = encoder.fit_transform(column_cat)

            new_columns = [f"{column}_{value}" for value in encoder.get_feature_names_out()]
            X[new_columns] = encoded_data

        X = X.drop(columns=self.columns)
        return X
    
class StandardColumnScaler(StandardScaler):
    def __init__(self, columns: list[str], copy:bool = True, with_mean: bool = True, with_std: bool = True) -> None:
        self.columns = columns
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)

    def fit(self, X: pd.DataFrame, y=None):
        self.scaler = super().fit(X[self.columns], y)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_subset = X[self.columns]
        X_subset = pd.DataFrame(columns=X_subset.columns, data=super().transform(X_subset[self.columns]))
        X[self.columns] = X_subset
        return X
    
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str], n_clusters: int=10, gamma: int=1.0, random_state:int = None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.columns = columns
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None, sample_weight=pd.Series):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state, init="k-means++")
        self.kmeans_.fit(X[self.columns], sample_weight=sample_weight)
        return self
    
    def transform(self, X: pd.DataFrame):
        return rbf_kernel(X[self.columns], self.kmeans_.cluster_centers_, gamma=self.gamma)


class ClusterSimilarityTransformer(ClusterSimilarity):

    def transform(self, X: pd.DataFrame):
        similarities =  super().transform(X)
        for i in range(1, self.n_clusters+1):
            X[f"Cluster_{i}"] = similarities[:,i-1]
        
        return X