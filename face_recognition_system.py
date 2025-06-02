"""
Face Recognition System
Core implementation for real-time face detection, recognition, and learning using OpenCV.
"""

import cv2
import numpy as np
import pickle
import os
from utils import get_user_input, save_face_data, load_face_data

class FaceRecognitionSystem:
    def __init__(self, data_file="face_data.pkl"):
        """
        Initialize the Face Recognition System.
        
        Args:
            data_file (str): Path to the file where face data will be stored
        """
        self.data_file = data_file
        self.known_faces = []  # Will store face features and names
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.is_trained = False
        
        # Load existing face data if available
        self.load_known_faces()
        
        # Initialize camera
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            raise Exception("Could not open camera. Please check your camera connection.")
        
        # Set camera properties for better performance
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"Loaded {len(self.known_faces)} known faces from storage.")

    def load_known_faces(self):
        """Load previously stored face data and names."""
        try:
            data = load_face_data(self.data_file)
            if data:
                self.known_faces = data.get('faces', [])
                if self.known_faces:
                    self.retrain_recognizer()
                print(f"Successfully loaded {len(self.known_faces)} known faces.")
            else:
                print("No existing face data found. Starting with empty database.")
        except Exception as e:
            print(f"Error loading face data: {str(e)}")
            print("Starting with empty face database.")

    def save_known_faces(self):
        """Save current face data and names to storage."""
        try:
            data = {
                'faces': self.known_faces
            }
            save_face_data(data, self.data_file)
            print(f"Face data saved successfully. Total faces: {len(self.known_faces)}")
        except Exception as e:
            print(f"Error saving face data: {str(e)}")

    def retrain_recognizer(self):
        """Retrain the face recognizer with current face data."""
        if not self.known_faces:
            self.is_trained = False
            return
        
        faces = []
        labels = []
        
        for i, face_data in enumerate(self.known_faces):
            for face_img in face_data['images']:
                faces.append(face_img)
                labels.append(i)
        
        if faces:
            self.face_recognizer.train(faces, np.array(labels))
            self.is_trained = True
            print(f"Face recognizer trained with {len(faces)} face samples.")

    def register_new_face(self, face_img, name):
        """
        Register a new face with the given name.
        
        Args:
            face_img: The face image to register
            name: The name for this face
        """
        # Check if this name already exists
        for face_data in self.known_faces:
            if face_data['name'].lower() == name.lower():
                # Add more samples for existing person
                face_data['images'].append(face_img)
                print(f"Added new sample for existing person: {name}")
                self.retrain_recognizer()
                self.save_known_faces()
                return
        
        # Create new face entry
        new_face = {
            'name': name,
            'images': [face_img]
        }
        self.known_faces.append(new_face)
        
        # Retrain recognizer and save
        self.retrain_recognizer()
        self.save_known_faces()
        print(f"New face registered: {name}")

    def get_user_input_for_face(self, frame):
        """Get user input for naming a new face."""
        print("\n" + "="*50)
        print("NEW FACE DETECTED!")
        print("="*50)
        
        # Show the current frame
        cv2.imshow('Face Recognition System', frame)
        cv2.waitKey(1)
        
        # Get name from user
        name = get_user_input("Please enter the name for this person: ")
        
        if name and name.strip():
            return name.strip()
        else:
            print("No name provided. Face not registered.")
            return None

    def process_frame(self, frame):
        """
        Process a single frame for face detection and recognition.
        
        Args:
            frame: The video frame to process
            
        Returns:
            processed_frame: Frame with face recognition annotations
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))
            
            name = "Unknown"
            confidence = 0
            
            if self.is_trained:
                # Try to recognize the face
                label, confidence = self.face_recognizer.predict(face_roi)
                
                # Set confidence threshold (lower is better for LBPH)
                if confidence < 100:  # You can adjust this threshold
                    if label < len(self.known_faces):
                        name = self.known_faces[label]['name']
                    else:
                        name = "Unknown"
                else:
                    name = "Unknown"
            
            # If face is unknown, ask for registration
            if name == "Unknown":
                user_name = self.get_user_input_for_face(frame)
                if user_name:
                    self.register_new_face(face_roi, user_name)
                    name = user_name
            
            # Draw face detection results
            self.draw_face_box(frame, x, y, w, h, name, confidence)
        
        return frame

    def draw_face_box(self, frame, x, y, w, h, name, confidence):
        """Draw bounding box and name on detected face."""
        # Choose color based on recognition status
        if name == "Unknown":
            color = (0, 0, 255)  # Red for unknown faces
        else:
            color = (0, 255, 0)  # Green for recognized faces
        
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Draw label background
        label_text = f"{name}"
        if confidence > 0:
            label_text += f" ({confidence:.1f})"
        
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x, y-30), (x+label_size[0]+10, y), color, cv2.FILLED)
        
        # Draw the name
        cv2.putText(frame, label_text, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run_real_time_recognition(self):
        """Main loop for real-time face recognition."""
        print("Face recognition system started. Press 'q' to quit.")
        print("Instructions:")
        print("- When a new face is detected, you'll be prompted to enter a name in the terminal")
        print("- Previously registered faces will be automatically recognized")
        
        while True:
            # Capture frame-by-frame
            ret, frame = self.video_capture.read()
            
            if not ret:
                print("Error: Could not read frame from camera.")
                break
            
            # Flip the frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame for face recognition
            frame = self.process_frame(frame)
            
            # Add system info overlay
            info_text = [
                f"Known faces: {len(self.known_faces)}",
                "Press 'q' to quit"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                y_offset += 25
            
            # Display the resulting frame
            cv2.imshow('Face Recognition System', frame)
            
            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        print("Shutting down face recognition system...")
        self.video_capture.release()
        cv2.destroyAllWindows()
        print("Cleanup completed.")

    def __del__(self):
        """Destructor to ensure cleanup."""
        if hasattr(self, 'video_capture'):
            self.cleanup()