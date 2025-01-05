import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt

class FaceRecognitionPCA:
    def __init__(self, n_components=100, confidence_threshold=100):
        self.n_components = n_components
        self.confidence_threshold = confidence_threshold
        self.scaler = StandardScaler()
        
    def load_images(self, data_path):
        """Load images from the specified directory."""
        faces = []
        labels = []
        for person_id in os.listdir(data_path):
            person_dir = os.path.join(data_path, person_id)
            if os.path.isdir(person_dir):
                for image_file in os.listdir(person_dir):
                    if image_file.endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(person_dir, image_file)
                        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (100, 100))  # Resize to standard size
                            faces.append(img.flatten())
                            labels.append(int(person_id))
        
        return np.array(faces), np.array(labels)
    
    def train(self, X, y):
        """Train the PCA model with face data."""
        if len(X) < 2:
            raise ValueError("Need at least 2 images in the dataset for training. Please add more images to your face_dataset directory.")
            
        # Adjust n_components to be at most the number of samples - 1
        n_components = min(self.n_components, len(X) - 1)
        self.pca = PCA(n_components=n_components)
        
        # Standardize the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA
        self.pca.fit(X_scaled)
        
        # Transform data to eigenface space
        self.faces_transformed = self.pca.transform(X_scaled)
        self.labels = y
        
        # Calculate average confidence score for known faces
        self.avg_confidence = self._calculate_avg_confidence()
        
    def _calculate_avg_confidence(self):
        """Calculate average confidence score for known faces."""
        confidences = []
        for i, face in enumerate(self.faces_transformed):
            # Calculate distances to all other faces
            distances = np.linalg.norm(self.faces_transformed - face.reshape(1, -1), axis=1)
            # Exclude distance to self (which would be 0)
            distances = distances[distances > 0]
            if len(distances) > 0:
                confidences.append(np.min(distances))
        return np.mean(confidences) if confidences else 100
        
    def recognize(self, face_image):
        """Recognize a face by finding the nearest neighbor in eigenface space."""
        # Preprocess the input image
        face_flattened = face_image.flatten().reshape(1, -1)
        face_scaled = self.scaler.transform(face_flattened)
        
        # Project into eigenface space
        face_projected = self.pca.transform(face_scaled)
        
        # Find nearest neighbor
        distances = np.linalg.norm(self.faces_transformed - face_projected, axis=1)
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        # If confidence score is too high (distance is too large), return Unknown
        if min_distance > self.confidence_threshold:
            return "Unknown", min_distance
            
        return self.labels[min_idx], min_distance
    
    def visualize_eigenfaces(self, n_components=5):
        """Visualize the first n eigenfaces."""
        # Adjust n_components to not exceed what we have
        n_components = min(n_components, len(self.pca.components_))
        
        fig, axes = plt.subplots(1, n_components, figsize=(2*n_components, 2))
        # Handle case where there's only one component
        if n_components == 1:
            axes = [axes]
            
        for i in range(n_components):
            eigenface = self.pca.components_[i].reshape(100, 100)
            axes[i].imshow(eigenface, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Eigenface {i+1}')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize the face recognition system
    face_recognition = FaceRecognitionPCA(n_components=50)
    
    # You'll need to specify your dataset path here
    data_path = "face_dataset"
    
    print("Please create a 'face_dataset' directory with subdirectories for each person")
    print("Each person's subdirectory should contain their face images")
    print("Example structure:")
    print("face_dataset/")
    print("├── 1/")
    print("│   ├── image1.jpg")
    print("│   └── image2.jpg")
    print("└── 2/")
    print("    ├── image1.jpg")
    print("    └── image2.jpg") 