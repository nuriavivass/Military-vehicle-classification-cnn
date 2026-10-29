"""
Nuria
Script to prepare the dataset
"""
import os
import numpy as np
import cv2
from PIL import Image
import random

root_path = "/your/path/to/dataset"

# Resize image 120x120
RESIZE_VALUE = 120
#number of augmentationxs
NUM_AUGMENTATIONS = 5

#First step, we load the labels with a TXT

def load_class_names():
    types_path = os.path.join(root_path, "TYPE-NAMES.txt")
    vehicle_types = []
    with open(types_path, "r") as f:
        for line in f.read().split():
            clean_name = line.replace('"', '').strip()
            if clean_name:
                vehicle_types.append(clean_name)

    print(f" {len(vehicle_types)} labels loaded")
    print(f"   labels: {', '.join(vehicle_types)}")
    return vehicle_types



# Second step AUGMENTATION-Function (Create NEW image)

def augment_image(img):

    # Random rotation(0-360 grados)
    angle = random.uniform(0, 360)
    img = img.rotate(angle, fillcolor=(0, 0, 0))

    # Zoom random (0.8x - 1.2x)
    zoom_factor = random.uniform(0.8, 1.2)
    w, h = img.size
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # This is to keep the size
    if zoom_factor < 1.0:
        new_img = Image.new('RGB', (w, h), (0, 0, 0))
        offset = ((w - new_w) // 2, (h - new_h) // 2)
        new_img.paste(img, offset)
        img = new_img
    else:
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        img = img.crop((left, top, left + w, top + h))

    # Flip horizontal
    if random.random() > 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    return img

  #Third step, this function use de augmentantion function with the real images

def augment_images(vehicle_types):

    for vehicle_class in vehicle_types:
        class_path = os.path.join(root_path, vehicle_class)

        if not os.path.exists(class_path):
            
            continue

        original_images = [f for f in os.listdir(class_path)
                          if f.endswith(('.jpg', '.jpeg', '.png'))
                          and not f.startswith('aug_')]

        print(f"\n {vehicle_class}: {len(original_images)} original photos")

        for img_name in original_images:
            try:
                
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                original_size = img.size
                img = img.resize((RESIZE_VALUE, RESIZE_VALUE), Image.Resampling.LANCZOS)
                img.save(img_path, 'JPEG', quality=95)
                base_name = os.path.splitext(img_name)[0]
                for i in range(NUM_AUGMENTATIONS):
                    augmented = augment_image(img.copy())
                    aug_name = f"aug_{base_name}_{i+1}.jpg"
                    aug_path = os.path.join(class_path, aug_name)
                    augmented.save(aug_path, 'JPEG', quality=95)
                idx = original_images.index(img_name)
                if (idx + 1) % 10 == 0:
                    print(f"   Processed {idx + 1}/{len(original_images)} ...")

            except Exception as e:
                print(f"    Error in {img_name}: {e}")

        all_images = [f for f in os.listdir(class_path)
                     if f.endswith(('.jpg', '.jpeg', '.png'))]

        verified = True
        for img_file in all_images[:5]:  
            img_check = Image.open(os.path.join(class_path, img_file))
            if img_check.size != (RESIZE_VALUE, RESIZE_VALUE):
                print(f"    {img_file} has size {img_check.size} {RESIZE_VALUE}×{RESIZE_VALUE})")
                verified = False

        if verified:
            print(f"    Total: {len(all_images)} images")
        else:
            print(f"     Total: {len(all_images)} images")

    
    print(" AUGMENTATION COMPLETE")
    

# ============================================================================
# Step 3: Create DATASET 
# ============================================================================

def create_training_data(vehicle_types):

    images_list = []
    labels_list = []
    size_errors = []

    for vehicle_class in vehicle_types:
        class_path = os.path.join(root_path, vehicle_class)
        class_num = vehicle_types.index(vehicle_class)

        if not os.path.exists(class_path):
            continue

        images = [f for f in os.listdir(class_path)
                 if f.endswith(('.jpg', '.jpeg', '.png'))]

        print(f"\n {vehicle_class} (ID {class_num}): {len(images)} image")

        for img_name in images:
            try:
                # load image
                img_path = os.path.join(class_path, img_name)
                image = Image.open(img_path)

                
                if image.size != (RESIZE_VALUE, RESIZE_VALUE):
                    size_errors.append((img_name, image.size))
                    print(f"     {img_name}:  {image.size} : {RESIZE_VALUE}×{RESIZE_VALUE})")
                    
                    image = image.resize((RESIZE_VALUE, RESIZE_VALUE), Image.Resampling.LANCZOS)

                
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                
                img_array = np.array(image)

                
                if img_array.shape != (RESIZE_VALUE, RESIZE_VALUE, 3):
                    print(f"     {img_name}: shape incorrecto {img_array.shape}")
                    img_array = cv2.resize(img_array, (RESIZE_VALUE, RESIZE_VALUE))

                
                images_list.append(img_array)
                labels_list.append(class_num)

            except Exception as e:
                print(f"    Error  {img_name}: {e}")

    
    images_array = np.array(images_list, dtype=np.uint8)
    labels_array = np.array(labels_list, dtype=np.int32)

    
    print(" DATASET ")
    print(f"   Total image: {len(images_list)}")
    print(f"   Shape image: {images_array.shape}")
    print(f"   Shape labels: {labels_array.shape}")

    if size_errors:
        print(f"\n   {len(size_errors)} image with wrong size")
    else:
        print("\n All  image with correct  size (120×120×3)")

    # Save  file .npz
    output_file = os.path.join(root_path, "military_vehicles_dataset.npz")
    np.savez_compressed(output_file, images=images_array, labels=labels_array)

    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    

    return images_array, labels_array


# ============================================================================
# Principal funtion
# ============================================================================

def main():
    # Step 1: Load labels
    vehicle_types = load_class_names()

    # Step 2: Augmentation (resize to 120×120)
    augment_images(vehicle_types)

    # Step 3: Create and  save dataset
    images, labels = create_training_data(vehicle_types)

    # Final metrics
    print(" Final metrics")

    unique_labels, counts = np.unique(labels, return_counts=True)

    print(f"\n{'Clase':<15} {'ID':<5} {'Image':<10}")
    print("-" * 35)
    for label_id, count in zip(unique_labels, counts):
        class_name = vehicle_types[label_id]
        print(f"{class_name:<15} {label_id:<5} {count:<10}")
    print("-" * 35)
    print(f"{'TOTAL':<15} {'':<5} {len(images):<10}")

    print(f"\n Saved Data: {root_path}/military_vehicles_dataset.npz")
    print(f" Total images: {len(images)}")
    print(f" All  images: {RESIZE_VALUE}×{RESIZE_VALUE}×3")
    print(f" Labels: {len(vehicle_types)}")

if __name__ == "__main__":
    main()