# src/recipe_analysis.py

import pandas as pd
import numpy as np

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

    def run_nmf(self, n_components=5):
        """
        run NMF on binary columns and store components
        """


    
