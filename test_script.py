"""
Inference Script - Military Vehicle Classifier
Loads trained model and classifies images from a selected folder
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from tensorflow.keras.preprocessing import image

#Configuration

ROOT_PATH = "/content/drive/MyDrive/PROYECTO NURIA/dataset/"
MODEL_PATH = ROOT_PATH + "military_vehicles_model_augmented.keras"
TEST_IMAGES_PATH = ROOT_PATH + "test_images/"  # Folder with images to classify

RESIZE_VALUE = 120
NUM_CLASSES = 11

# Class names 
CLASS_NAMES = [
    "A400",
    "B2", 
    "BTR",
    "euro",
    "F35",
    "leopard2",
    "M1",
    "M109",
    "Mig29",
    "SU25",
    "T72"
]

def load_model():
    """Load the trained model"""
    
    print("\n" + "="*50)
    print(" LOADING MODEL")
    print("="*50)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    
    model = keras.models.load_model(MODEL_PATH)
    print(f" Model loaded from: {MODEL_PATH}")
    
    return model

def load_and_preprocess_image(img_path):
    """Load and preprocess a single image for prediction"""
    
    # Load image with target size
    img = image.load_img(img_path, target_size=(RESIZE_VALUE, RESIZE_VALUE))
    
    # Convert to array
    img_array = image.img_to_array(img)
    
    # Normalize [0-255] -> [0-1]
    img_array = img_array / 255.0
    
    # Add batch dimension (1, 120, 120, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img, img_array


def predict_image(model, img_path):
    """Make prediction for a single image"""
    
    # Load and preprocess
    original_img, processed_img = load_and_preprocess_image(img_path)
    
    # Predict
    predictions = model.predict(processed_img, verbose=0)
    
    # Get results
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    
    return original_img, predicted_class, confidence, predictions[0]


def classify_folder(model, folder_path):
    
    
    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # Get all image files
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f" No images found in: {folder_path}")
        return []
    
    print(f" Found {len(image_files)} images\n")
    
    results = []
    
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        
        try:
            original_img, predicted_class, confidence, all_probs = predict_image(model, img_path)
            
            result = {
                'filename': img_file,
                'predicted_class': predicted_class,
                'class_name': CLASS_NAMES[predicted_class],
                'confidence': confidence,
                'all_probabilities': all_probs,
                'image': original_img
            }
            results.append(result)
            
            print(f" {img_file}")
            print(f"   -> {CLASS_NAMES[predicted_class]} ({confidence:.2f}%)\n")
            
        except Exception as e:
            print(f" Error processing {img_file}: {str(e)}\n")
    
    return results



def visualize_results(results, max_images=12):
    """Display classification results in a grid"""
    
    print("\n" + "="*50)
    print(" VISUALIZING RESULTS")
    print("="*50)
    
    if not results:
        print(" No results to visualize")
        return
    
    # Limit number of images to display
    results_to_show = results[:max_images]
    n_images = len(results_to_show)
    
    # Calculate grid size
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    # Handle single row/column case
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, result in enumerate(results_to_show):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        ax.imshow(result['image'])
        ax.axis('off')
        
        # Color based on confidence
        color = 'green' if result['confidence'] > 70 else 'orange' if result['confidence'] > 50 else 'red'
        
        ax.set_title(f"{result['class_name']}\n{result['confidence']:.1f}%", 
                     fontsize=10, color=color, fontweight='bold')
    
    # Hide empty subplots
    for idx in range(n_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    save_path = ROOT_PATH + "classification_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f" Results saved at: {save_path}")
    
    plt.show()


def show_detailed_prediction(model, img_path):
    """Show detailed prediction with probability bar chart"""
    
    print("\n" + "="*50)
    print(" DETAILED PREDICTION")
    print("="*50)
    
    original_img, predicted_class, confidence, all_probs = predict_image(model, img_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Show image
    ax1.imshow(original_img)
    ax1.axis('off')
    ax1.set_title(f"Prediction: {CLASS_NAMES[predicted_class]}\nConfidence: {confidence:.2f}%",
                  fontsize=14, fontweight='bold')
    
    # Show probability bar chart
    colors = ['green' if i == predicted_class else 'steelblue' for i in range(NUM_CLASSES)]
    bars = ax2.barh(CLASS_NAMES, all_probs * 100, color=colors)
    ax2.set_xlabel("Probability (%)", fontsize=12)
    ax2.set_title("Class Probabilities", fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 100])
    
    # Add percentage labels
    for bar, prob in zip(bars, all_probs):
        if prob > 0.01:
            ax2.text(prob * 100 + 1, bar.get_y() + bar.get_height()/2, 
                    f'{prob*100:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return predicted_class, confidence


def print_summary(results):
    """Print classification summary"""
    
    print("\n" + "="*50)
    print(" CLASSIFICATION SUMMARY")
    print("="*50)
    
    if not results:
        print(" No results")
        return
    
    # Count predictions per class
    class_counts = {}
    for r in results:
        class_name = r['class_name']
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1
    
    print(f"\n Total images classified: {len(results)}")
    print(f"\n Predictions by class:")
    
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {class_name}: {count}")
    
    # Average confidence
    avg_confidence = np.mean([r['confidence'] for r in results])
    print(f"\n Average confidence: {avg_confidence:.2f}%")



def main():
    """Run inference on all images in the test folder"""
    
    print("\n" + "="*50)
    print(" MILITARY VEHICLE CLASSIFIER - INFERENCE")
    print("="*50)
    
    # 1. Load model
    model = load_model()
    
    # 2. Classify all images in folder
    results = classify_folder(model, TEST_IMAGES_PATH)
    
    # 3. Visualize results
    if results:
        visualize_results(results)
        print_summary(results)
    
    print("\n" + "="*50)
    print(" INFERENCE COMPLETED")
    print("="*50)
    
    return results


if __name__ == "__main__":
    results = main()