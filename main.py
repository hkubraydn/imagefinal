"""
Real-time Face Recognition System
Main entry point for the face recognition application.
"""

import sys
import cv2
from face_recognition_system import FaceRecognitionSystem

def main():
    """Main function to run the face recognition system."""
    print("=== Real-time Face Recognition System ===")
    print("Instructions:")
    print("- Press 'q' to quit the application")
    print("- When a new face is detected, you'll be prompted to enter a name")
    print("- Previously registered faces will be automatically recognized")
    print("\nStarting camera...")
    
    try:
        # Initialize the face recognition system
        face_system = FaceRecognitionSystem()
        
        # Start the real-time recognition
        face_system.run_real_time_recognition()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please make sure you have a working camera connected.")
    finally:
        print("Cleaning up...")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
