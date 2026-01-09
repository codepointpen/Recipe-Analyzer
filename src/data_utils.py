# src/recipe_analysis.py

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, NMF

class RecipeAnalysis:
    def __init__(self, file_path):
        """
        initialize with dataset path
        """
        self.recipes = pd.read_csv(file_path)
        self.binary_cols = []
        self.continuous_cols = []
        self.pca_components = None
        self.nmf_components = None
        self.lasso_coef_df = None
        self.cooc_graph = None

    def preprocess(self):
        """
        identify numeric, binary, and continuous columns
        """
        numeric_cols = self.recipes.select_dtypes(include=np.number).columns.tolist()
        self.binary_cols = [c for c in numeric_cols if self.recipes[c].dropna().isin([0,1]).all()]
        self.continuous_cols = list(set(numeric_cols) - set(self.binary_cols))
        return self.binary_cols, self.continuous_cols


    def run_pca(self, n_components=5):
        """
        run PCA on binary columns and store components
        """
        X = self.recipes[self.binary_cols].values
        pca = PCA(n_components=n_components)
        self.pca_components = pca.fit_transform(X)
        for i in range(n_components):
            self.recipes[f'pca_{i+1}'] = self.pca_components[:, i]
        return self.pca_components

    def run_nmf(self, n_components=5):
        """
        run NMF on binary columns and store components
        """
        X = self.recipes[self.binary_cols].values
        nmf = NMF(n_components=n_components, init='random', random_state=42)
        self.nmf_components = nmf.fit_transform(X)
        for i in range(n_components):
            self.recipes[f'nmf_{i+1}'] = self.nmf_components[:, i]
        return self.nmf_components


    
