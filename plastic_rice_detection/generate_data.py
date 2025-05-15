import numpy as np
import os
import cv2

# Create directories for real and plastic rice images
os.makedirs('data/real_rice', exist_ok=True)
os.makedirs('data/plastic_rice', exist_ok=True)

# Generate synthetic images for real rice
for i in range(10):
    # Create a random image with rice-like texture
    img = np.random.normal(128, 30, (224, 224, 3)).astype(np.uint8)
    cv2.imwrite(f'data/real_rice/real_rice_{i}.jpg', img)

# Generate synthetic images for plastic rice
for i in range(10):
    # Create a random image with plastic-like texture
    img = np.random.normal(200, 20, (224, 224, 3)).astype(np.uint8)
    cv2.imwrite(f'data/plastic_rice/plastic_rice_{i}.jpg', img)

print('Data generation completed.') 