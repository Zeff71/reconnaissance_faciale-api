import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# 🔁 Charger les embeddings et labels (générés lors de l'entraînement)
with open('C:/Users/Zeff/reconnaissance_faciale/app/output/embeddings_labels.npz', 'rb') as f:
    data = np.load(f)
    embeddings = data['embeddings']
    labels = data['labels']

# 🔁 Charger le classificateur entraîné
with open('C:/Users/Zeff/reconnaissance_faciale/app/output/first_classifier.pkl', 'rb') as f:
    (model, class_names) = pickle.load(f)

# ⚙️ Découpage en jeu d'entraînement/test
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, stratify=labels, random_state=42
)

# 🔍 Prédictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Précision (accuracy) du classificateur : {accuracy * 100:.2f}%")

# 📊 Matrice de confusion
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.title("Matrice de confusion")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.tight_layout()

# 💾 Sauvegarde dans le conteneur
plt.savefig('output/confusion_matrix.png')
print("📁 La matrice de confusion a été enregistrée dans output/confusion_matrix.png")
