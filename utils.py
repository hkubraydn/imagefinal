"""
Utility functions for the face recognition system.
Contains helper functions for data storage, user input, and file operations.
"""

import pickle
import os
import sys
from threading import Thread
import time

def save_face_data(data, filename):
    """
    Save face data to a pickle file.
    
    Args:
        data (dict): Dictionary containing face encodings and names
        filename (str): Path to the file where data will be saved
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save data to pickle file
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Face data saved to {filename}")
        
    except Exception as e:
        raise Exception(f"Error saving face data: {str(e)}")

def load_face_data(filename):
    """
    Load face data from a pickle file.
    
    Args:
        filename (str): Path to the file containing face data
        
    Returns:
        dict or None: Dictionary containing face encodings and names, or None if file doesn't exist
    """
    try:
        if not os.path.exists(filename):
            print(f"Face data file {filename} not found. Starting with empty database.")
            return None
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        # Validate data structure
        if not isinstance(data, dict) or 'encodings' not in data or 'names' not in data:
            print("Invalid face data format. Starting with empty database.")
            return None
        
        if len(data['encodings']) != len(data['names']):
            print("Inconsistent face data. Starting with empty database.")
            return None
        
        return data
        
    except Exception as e:
        print(f"Error loading face data: {str(e)}")
        return None

def get_user_input(prompt):
    """
    Get user input with a prompt.
    This function handles input in a way that works with the OpenCV window system.
    
    Args:
        prompt (str): The prompt to display to the user
        
    Returns:
        str: The user's input
    """
    print(f"\n{prompt}")
    print("(Enter your response in the terminal)")
    
    try:
        # Use input() for getting user input
        user_input = input(">>> ").strip()
        return user_input
    except KeyboardInterrupt:
        print("\nInput cancelled.")
        return ""
    except Exception as e:
        print(f"Error getting user input: {str(e)}")
        return ""

def validate_name(name):
    """
    Validate a person's name.
    
    Args:
        name (str): The name to validate
        
    Returns:
        bool: True if the name is valid, False otherwise
    """
    if not name or not name.strip():
        return False
    
    # Remove extra whitespace
    name = name.strip()
    
    # Check if name contains only valid characters (letters, spaces, hyphens, apostrophes)
    valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -'")
    if not all(char in valid_chars for char in name):
        return False
    
    # Check length constraints
    if len(name) < 1 or len(name) > 50:
        return False
    
    return True

def create_backup(filename):
    """
    Create a backup of the face data file.
    
    Args:
        filename (str): Path to the file to backup
    """
    if os.path.exists(filename):
        backup_filename = f"{filename}.backup"
        try:
            import shutil
            shutil.copy2(filename, backup_filename)
            print(f"Backup created: {backup_filename}")
        except Exception as e:
            print(f"Warning: Could not create backup: {str(e)}")

def print_system_info():
    """Print system information for debugging purposes."""
    print("\n=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        print("OpenCV not found")

    
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError:
        print("NumPy: Not found")
    
    print("="*30)

def check_camera_availability():
    """
    Check if camera is available.
    
    Returns:
        bool: True if camera is available, False otherwise
    """
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            return ret
        return False
    except Exception:
        return False

def format_face_data_summary(known_faces):
    """
    Format a summary of known faces for display.
    
    Args:
        known_faces (list): List of known face names
        
    Returns:
        str: Formatted summary string
    """
    if not known_faces:
        return "No faces registered yet."
    
    summary = f"Registered faces ({len(known_faces)}):\n"
    for i, name in enumerate(known_faces, 1):
        summary += f"  {i}. {name}\n"
    
    return summary
