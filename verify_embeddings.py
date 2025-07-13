import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Chargement du fichier .npz
data = np.load("output/embeddings_labels.npz")
embeddings = data['embeddings']
labels = data['labels']

# Informations générales
print("Shape des embeddings:", embeddings.shape)  # Doit être (n_samples, 128)
print("Nombre de labels:", len(labels))
print("Exemples de labels:", labels[:15])

# Comparaison entre deux embeddings
if embeddings.shape[0] >= 2:
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])
    print("Similarité cosine entre les 2 premiers embeddings:", sim[0][0])
else:
    print("Pas assez d'embeddings pour comparer.")
