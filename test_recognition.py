from face_recognition_pca import FaceRecognitionPCA
import cv2
import os
from tkinter import filedialog
import tkinter as tk
from collections import deque
import numpy as np
import time

class ConfidenceTracker:
    def __init__(self, window_size=30):
        self.scores = {}  # Dictionary to store scores for each person
        self.window_size = window_size
    
    def update(self, person_id, confidence):
        if person_id not in self.scores:
            self.scores[person_id] = deque(maxlen=self.window_size)
        self.scores[person_id].append(confidence)
    
    def get_average(self, person_id):
        if person_id in self.scores and len(self.scores[person_id]) > 0:
            return np.mean(self.scores[person_id])
        return None

def main():
    # Create face dataset directory if it doesn't exist
    if not os.path.exists('face_dataset'):
        os.makedirs('face_dataset/1')
        os.makedirs('face_dataset/2')
        print("Created face_dataset directory structure")
        print("Please add face images to:")
        print("face_dataset/1/ for person 1")
        print("face_dataset/2/ for person 2")
        return

    # Initialize the face recognition system with confidence threshold
    face_recognition = FaceRecognitionPCA(n_components=50, confidence_threshold=100)
    
    # Load the training images
    print("Loading training images...")
    faces, labels = face_recognition.load_images('face_dataset')
    
    if len(faces) == 0:
        print("No images found! Please add images to the face_dataset directory")
        return
        
    print(f"Loaded {len(faces)} images")
    
    # Train the model
    print("Training the model...")
    face_recognition.train(faces, labels)
    
    # Visualize the eigenfaces
    print("Displaying eigenfaces...")
    face_recognition.visualize_eigenfaces(5)
    
    # Create root window and hide it
    root = tk.Tk()
    root.withdraw()
    
    # Ask user for choice
    print("\nChoose testing method:")
    print("1. Select an image file")
    print("2. Use webcam")
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        # Open file dialog for image selection
        test_img_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png"),
                ("All files", "*.*")
            ]
        )
        
        if test_img_path:
            # Test with provided image
            img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print("Error: Could not load the image")
                return
            img = cv2.resize(img, (100, 100))
            person_id, confidence = face_recognition.recognize(img)
            
            # Display the test image with result
            img_display = cv2.imread(test_img_path)
            label = f"Person {person_id}" if person_id != "Unknown" else "Unknown"
            color = (0, 255, 0) if person_id != "Unknown" else (0, 0, 255)  # Green for known, Red for unknown
            
            cv2.putText(img_display, label, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(img_display, f"Conf: {confidence:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow('Recognition Result', img_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            print(f"Recognized as {label} with confidence score {confidence:.2f}")
    else:
        # Test with webcam
        print("Starting webcam... Press 'q' to quit")
        print("\nConfidence Score Legend:")
        print("- Lower scores indicate better matches")
        print("- Scores > 100 are considered 'Unknown'")
        print("- Showing running average over last 30 frames")
        print("\nReal-time Confidence Scores:")
        
        cap = cv2.VideoCapture(0)
        confidence_tracker = ConfidenceTracker(window_size=30)
        last_print_time = time.time()
        print_interval = 0.5  # Print every 0.5 seconds
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not access webcam")
                break
                
            # Convert to grayscale and resize
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces_rect = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            current_time = time.time()
            should_print = current_time - last_print_time >= print_interval
            
            for (x, y, w, h) in faces_rect:
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (100, 100))
                person_id, confidence = face_recognition.recognize(face_img)
                
                # Update confidence tracker
                confidence_tracker.update(person_id, confidence)
                
                # Set color based on recognition result
                color = (0, 255, 0) if person_id != "Unknown" else (0, 0, 255)  # Green for known, Red for unknown
                label = f"Person {person_id}" if person_id != "Unknown" else "Unknown"
                
                # Draw rectangle and labels
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(frame, f"Conf: {confidence:.2f}", (x, y+h+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Print confidence scores to terminal
                if should_print:
                    avg_confidence = confidence_tracker.get_average(person_id)
                    if avg_confidence is not None:
                        print(f"\r{label} - Current: {confidence:.2f}, Average: {avg_confidence:.2f}    ", end="", flush=True)
            
            if should_print:
                last_print_time = current_time
            
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        print("\nWebcam session ended")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 