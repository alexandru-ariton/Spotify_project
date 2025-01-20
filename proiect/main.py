import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

from functii import *
from grafice import *

set_date = pd.read_csv("spotify-2023.csv", index_col=0)
variabile = list(set_date)[1:]
valori_lipsa = set_date.isna().any().any()
if valori_lipsa:
    nan_replace(set_date)


x = set_date[variabile].values
metoda = "ward"
h = linkage(x, method=metoda)
n = len(set_date)
clusteri_singleton = leaves_list(h)
k_opt, threshold_opt, p_opt = partitie(h)
silhouette_opt = silhouette_score(x, p_opt)
index_silhouette_opt = silhouette_samples(x, p_opt)
culori = generare_rampa("rainbow", k_opt)


culori_instante = []
for i in range(n):
    index_cluster = int(p_opt[clusteri_singleton[i]][1:])-1
    culori_instante.append(culori[index_cluster])

culori_dendrograma = unique(culori_instante)

plot_ierarhie(h, threshold_opt,
              "Plot ierarhie - partitia optimala. Scor Silhouette:" + str(silhouette_opt),
              k_opt, set_date.index, culori=culori_dendrograma)
plt.savefig('out/dendograma')
show()


file_path = 'spotify-2023.csv'
spotify_data = pd.read_csv(file_path)


categorical_features = ['track_name', 'artist(s)_name']
numerical_features = spotify_data.columns.difference(categorical_features)


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


processed_data = preprocessor.fit_transform(spotify_data)


dense_data = processed_data.toarray()


pca = PCA(n_components=2)
pca_data = pca.fit_transform(dense_data)


explained_variance = pca.explained_variance_ratio_


pca_df = pd.DataFrame(data=pca_data, columns=['Principal Component 1', 'Principal Component 2'])
plt.figure(figsize=(10, 8))
plt.scatter(pca_df['Principal Component 1'], pca_df['Principal Component 2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of Spotify Dataset')
plt.grid()
plt.savefig('out/grafic_PCA')
show()

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, color='g', label='Varianța individuală')
plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', color='r', label='Varianța cumulativă')
plt.xlabel('Numărul componentelor principale')
plt.ylabel('Proporția varianței explicată')
plt.title('Graficul scree pentru PCA')
plt.legend(loc='best')
plt.savefig('out/grafic_scree')
show()


pca_components = pca.components_
scores = pca_data[:, :2]


plt.figure(figsize=(10, 6))
plt.scatter(scores[:, 0], scores[:, 1], alpha=0.7, color='blue')
for i in range(pca_components.shape[1]):
    plt.arrow(0, 0, pca_components[0, i]*max(scores[:, 0]), pca_components[1, i]*max(scores[:, 1]), color='r', alpha=0.5)
    if i < len(numerical_features):
        plt.text(pca_components[0, i]*max(scores[:,0]), pca_components[1, i]*max(scores[:, 1]), numerical_features[i], color='black')

plt.xlabel('Componenta principală 1')
plt.ylabel('Componenta principală 2')
plt.title('Biplot pentru primele două componente principale')
plt.grid()
plt.savefig('out/grafic_biplot')
show()