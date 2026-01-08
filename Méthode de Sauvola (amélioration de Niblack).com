from google.colab import files
import cv2
import numpy as np
import matplotlib.pyplot as plt



# Charger l'image (remplace le nom si nécessaire)
image = cv2.imread('doc4 (1).jpeg', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image non trouvée")
    exit()

# Paramètres Sauvola
window = 15
k = 0.4
R = 128

img_float = image.astype(np.float32)
mean = cv2.boxFilter(img_float, -1, (window, window))
mean_sq = cv2.boxFilter(img_float * img_float, -1, (window, window))
std = np.sqrt(mean_sq - mean * mean)

threshold = mean * (1 + k * ((std / R) - 1))
binary = np.zeros_like(image)
binary[img_float >= threshold] = 255

plt.imshow(binary, cmap='gray')
plt.title("Résultat Sauvola")
plt.axis('off')
plt.show()
