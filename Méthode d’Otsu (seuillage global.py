import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger l'image importée (remplace par le vrai nom si nécessaire)
image = cv2.imread('doc4 (1).jpeg', cv2.IMREAD_GRAYSCALE)

# Vérification
if image is None:
    print("Erreur : image non trouvée")
    exit()

# ===== Paramètres Niblack =====
window_size = 15   # Taille de la fenêtre locale (impair)
k = -0.2           # Paramètre Niblack

# Conversion en float pour calcul
img_float = image.astype(np.float32)

# Moyenne locale
mean = cv2.boxFilter(img_float, -1, (window_size, window_size))

# Écart-type local
mean_sq = cv2.boxFilter(img_float * img_float, -1, (window_size, window_size))
std_dev = np.sqrt(mean_sq - mean * mean)

# Seuil Niblack
threshold_niblack = mean + k * std_dev

# Application du seuillage
niblack_binary = np.zeros_like(image)
niblack_binary[img_float >= threshold_niblack] = 255
niblack_binary = niblack_binary.astype(np.uint8)

# ===== Affichage =====
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Image originale")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(niblack_binary, cmap='gray')
plt.title(f"Résultat - Niblack (k={k})")
plt.axis('off')

plt.tight_layout()
plt.show()
