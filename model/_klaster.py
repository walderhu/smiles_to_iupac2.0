import os
import warnings
from csv import QUOTE_ALL
from io import StringIO
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chython import MoleculeContainer, smiles
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from _model import batch_reader

warnings.filterwarnings("ignore", category=FutureWarning)


dirname = '.'
test_dirname = '__test__'
train_dirname = '__train__'
os.makedirs(test_dirname, exist_ok=True)
os.makedirs(train_dirname, exist_ok=True)
def test_label(file): return join(test_dirname, f'test_{file}')
def train_label(file): return join(train_dirname, f'train_{file}')


def smiles_to_fp(smiles_str):
    if mol := smiles(smiles_str):
        fp = MoleculeContainer.morgan_fingerprint(mol, min_radius=2, number_active_bits=2048)
        return np.array(fp, dtype=int)


class Klaster:
    def __init__(self, df: pd.DataFrame, max_clusters=50):
        self.df = df
        fps = df['SMILES'].apply(smiles_to_fp).dropna().to_list()
        self.X = np.array(fps)
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

        self.pca = PCA(n_components=0.95)
        self.X_reduced = self.pca.fit_transform(self.X_scaled)

        self.n_clusters = self._estimate_clusters(max_k=max_clusters)
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, batch_size=1024)

        self.clusters = self.kmeans.fit_predict(self.X_reduced)
        self.name, self.reducer = "t-SNE", TSNE(n_components=2, random_state=42, perplexity=30)
        self.embedding = self.reducer.fit_transform(self.X_reduced)

    def _estimate_clusters(self, max_k=15, min_k=5, step=1, plot=False):
        test_size = min(10000, len(self.X_reduced))
        X_test = self.X_reduced[:test_size]
        inertias = []
        k_values = range(min_k, max_k + 1, step)
        for k in k_values:
            kmeans = MiniBatchKMeans(n_clusters=k, batch_size=1024, random_state=42)
            kmeans.fit(X_test)
            inertias.append(kmeans.inertia_)

        deltas = np.diff(inertias)
        double_deltas = np.diff(deltas)
        optimal_idx = np.argmax(np.abs(double_deltas)) + 1
        optimal_k = k_values[optimal_idx]

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(k_values, inertias, 'bo-')
            plt.axvline(x=optimal_k, color='r', linestyle='--')
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('Inertia')
            plt.title('Elbow Method')
            plt.savefig('num_clasters.png', dpi=300)
            plt.close()
        return optimal_k

    def handle_unique_indices(self, *, percentage=5):
        distances = self.kmeans.transform(self.X_reduced)
        min_distances = np.min(distances, axis=1)
        n_select = max(1, int(len(self.X_reduced) * percentage / 100))
        unique_indices = np.argsort(min_distances)[-n_select:]
        return unique_indices

    def save(self, *, filename='unique_molecules.csv'):
        unique_indices = self.handle_unique_indices()
        unique_molecules_df = self.df.iloc[unique_indices][['cid', 'IUPAC Name', 'SMILES']]
        print(f"\n5% самых уникальных молекул ({len(unique_indices)} из {len(self.X_scaled)}):")
        unique_molecules_df.to_csv(filename, index=False, encoding='utf-8')

    def draw(self, *, filename=None):
        plt.figure(figsize=(10, 8))
        plt.scatter(self.embedding[:, 0], self.embedding[:, 1], c=self.clusters, cmap='tab20', s=30, alpha=0.7)
        plt.title(f'Кластеризация молекул ({self.name})', fontsize=14)
        plt.xlabel(f'{self.name} 1', fontsize=12)
        plt.ylabel(f'{self.name} 2', fontsize=12)
        plt.colorbar(label='Кластер')
        plt.grid(alpha=0.2)
        sample_indices = np.random.choice(len(self.embedding), size=min(15, len(self.embedding)), replace=False)
        for i in sample_indices:
            plt.annotate(self.df.iloc[i]['cid'], (self.embedding[i, 0], self.embedding[i, 1]),
                         textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)
        plt.tight_layout()
        if not filename:
            filename = f'clusters_{self.name.lower()}.png'
        plt.savefig(filename, dpi=300)
        plt.close()


def csv_format(num_line):
    return dict(index=False, encoding='utf-8', mode='w' if num_line == 1 else 'a',
                header=(num_line == 1), sep=';', escapechar='\\', quoting=QUOTE_ALL, quotechar='"')


def main():
    files = [f for f in os.listdir(dirname) if f.endswith('.csv') and not f.startswith('_')]
    valid_repr = []
    for file in files:
        for num_line, batch in enumerate(batch_reader(join(dirname, file), batch_size=1e4), start=1):
            df = pd.read_csv(StringIO(batch), delimiter=';', encoding='utf-8').dropna()
            klaster = Klaster(df)
            test_indices = klaster.handle_unique_indices()
            test_molecules_df, train_molecules_df = df.iloc[test_indices], df.drop(test_indices)
            valid_repr.append(test_molecules_df.sample(frac=0.1, random_state=42))

            test_molecules_df.to_csv(test_label(file), **csv_format(num_line))
            train_molecules_df.to_csv(train_label(file), **csv_format(num_line))

    valid_repr = pd.concat(valid_repr, ignore_index=True)
    valid_repr.to_csv(f'_validation.csv', **csv_format(num_line=1))


if __name__ == '__main__':
    main()
