import cv2
import matplotlib.pyplot as plt

# Charger l'image importée (remplace par le vrai nom si nécessaire)
image = cv2.imread('doc4 (1).jpeg', cv2.IMREAD_GRAYSCALE)

# Vérification
if image is None:
    print("Erreur : image non trouvée")
    exit()

# --- Seuillage global : Otsu ---
seuil, image_otsu = cv2.threshold(
    image, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print("Seuil calculé par Otsu :", seuil)

# --- Affichage ---
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Image originale")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_otsu, cmap='gray')
plt.title("Résultat - Seuillage Otsu")
plt.axis('off')

plt.tight_layout()
plt.show()
