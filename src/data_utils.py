# src/recipe_analysis.py

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, NMF
import networkx as nx
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import seaborn as sns

class RecipeAnalysis:
    def __init__(self, file_path):
        """
        initialize with dataset path
        """
        self.recipes = pd.read_csv(file_path, encoding="utf-8-sig")
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
    
    
    def build_coocurrence_graph(self, threshold=100):
        """
        build a co-occurrence network from binary features
        """
        cooc = self.recipes[self.binary_cols].T.dot(self.recipes[self.binary_cols])
        np.fill_diagonal(cooc.values, 0)
        cooc_threshold = cooc.stack()[cooc.stack() >= threshold]
        self.cooc_graph = nx.from_pandas_edgelist(
            cooc_threshold.reset_index(), source='level_0', target='level_1', edge_attr=0
        )
        return self.cooc_graph
        

    def run_lasso(self, target_col='rating', cv=5):
        """
        fit Lasso regression and store coefficients
        """
        X = self.recipes[self.binary_cols].values
        y = self.recipes[target_col].values
        model = LassoCV(cv=cv).fit(X, y)
        self.lasso_coef_df = pd.DataFrame({
            'tag': self.binary_cols,
            'coef': model.coef_
        }).sort_values('coef', ascending=False)
        return self.lasso_coef_df
    
    def plot_top_tags(self, top_n=10):
        """
        plot top positive and negative tags from regression
        """
        if self.lasso_coef_df is None:
            raise ValueError("Run run_lasso() first!")
        top_pos = self.lasso_coef_df.sort_values("coef", ascending=False).head(top_n)
        top_neg = self.lasso_coef_df.sort_values("coef").head(top_n)
        plt.figure(figsize=(10,6))
        sns.barplot(x="coef", y="tag", data=pd.concat([top_pos, top_neg]))
        plt.title("Top Positive and Negative Tags for Recipe Ratings")
        plt.xlabel("Coefficient (Impact on Rating)")
        plt.ylabel("Tag")
        plt.show()
        
    def plot_cooc_network(self, top_n_edges=50):
        """
        plot the top co-occurring ingredient/metadata tags in the network
        """
        if self.cooc_graph is None:
            raise ValueError("Run build_coocurrence_graph() first!")

        # sort edges by weight 
        edges = sorted(self.cooc_graph.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
        top_edges = edges[:top_n_edges]

        G = nx.Graph()
        G.add_edges_from([(u, v, {'weight': w['weight']}) for u, v, w in top_edges])

        # draw the network
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, seed=42) 
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
        nx.draw_networkx_edges(G, pos, width=[d['weight']*0.05 for (_, _, d) in G.edges(data=True)])
        nx.draw_networkx_labels(G, pos, font_size=10)
        plt.title("Top Co-occurring Recipe Tags Network")
        plt.axis('off')
        plt.show()
        

    
