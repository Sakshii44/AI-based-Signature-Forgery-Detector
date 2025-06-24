import cv2
import os

# Input and output folder paths
input_base = 'signature_dataset'
output_base = 'processed_dataset'

# Target image size
IMAGE_SIZE = (220, 155)

# Ensure output folders exist
for split in ['train', 'val', 'test']:
    for label in ['genuine', 'forged']:
        os.makedirs(os.path.join(output_base, split, label), exist_ok=True)

# Preprocessing function
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Grayscale
    if img is None:
        return None
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Binarize
    img_resized = cv2.resize(img_bin, IMAGE_SIZE)  # Resize
    return img_resized

# Process images
for split in ['train', 'val', 'test']:
    for label in ['genuine', 'forged']:
        input_folder = os.path.join(input_base, split, label)
        output_folder = os.path.join(output_base, split, label)

        for filename in os.listdir(input_folder):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            processed_img = preprocess_image(input_path)
            if processed_img is not None:
                cv2.imwrite(output_path, processed_img)

print("✅ Preprocessing complete. Processed images saved to 'processed_dataset'.")
