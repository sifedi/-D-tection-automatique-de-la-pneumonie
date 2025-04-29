import numpy as np
import os
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

# Charger le modèle préentraîné
model = load_model("C:\\Users\\USER\\Desktop\\i2S1\\deep learning\\projetia\\best_model.keras")

# Fonction de prédiction
def predict(model, img):
    # Prétraitement des images
    x = img_to_array(img)
    x = x / 255.0  # Normalisation des pixels pour correspondre au modèle précédent
    x = np.expand_dims(x, axis=0)  # Ajouter une dimension batch
    preds = model.predict(x)  # Faire une prédiction
    return preds[0]

# Initialisation des fichiers de sortie
text_file1 = open("C:\\Users\\USER\\Desktop\\i2S1\\deep learning\\projetia\\test.txt", "w")
text_file2 = open("C:\\Users\\USER\\Desktop\\i2S1\\deep learning\\projetia\\test1.txt", "w")

# Variables pour les métriques
img_paths = {
    "NORMAL": "E:\\projet ia\\archive\\chest_xray\\train\\NORMAL",
    "PNEUMONIA": "E:\\projet ia\\archive\\chest_xray\\train\\PNEUMONIA",
}

results = {}

# Boucle sur les deux catégories (NORMAL et PNEUMONIA)
for label, img_path in img_paths.items():
    vrai = 0
    totale = 0
    print(f"Processing images for label: {label}")

    for img_name in sorted(os.listdir(img_path)):
        # Vérifier les extensions d'images valides
        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue

        filepath = os.path.join(img_path, img_name)
        img = load_img(filepath, target_size=(224, 224))  # Taille adaptée à EfficientNetB0
        totale += 1
        preds = predict(model, img)  # Prédictions pour l'image

        # Vérifier la classe prédite
        if (label == "NORMAL" and preds[0] >= 0.5) or (label == "PNEUMONIA" and preds[1] >= 0.5):
            vrai += 1
        else:
            if label == "NORMAL":
                text_file1.write(filepath + "\n")
            else:
                text_file2.write(filepath + "\n")

        print(f"{img_name}: {preds}")

    # Calculer l'accuracy pour la catégorie
    accuracy = (vrai / totale) * 100 if totale > 0 else 0
    results[label] = accuracy
    print(f"Accuracy for {label}: {accuracy:.2f}%")

# Calculer l'accuracy moyenne
acc_moyenne = np.mean(list(results.values()))
print(f"Accuracy moyenne: {acc_moyenne:.2f}%")

# Fermer les fichiers
text_file1.close()
text_file2.close()
