import calendar
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go
from streamlit_option_menu import option_menu


from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import plotly
import plotly.graph_objects as go








hide_st_style = """<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;} </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)



import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import tempfile

def load_model(model_path):
    model = YOLO(model_path)
    return model

def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
        conf (float): Confidence threshold for object detection.
        model (YOLO): An instance of the `YOLOv8` class containing the YOLOv8 model.
        st_frame (Streamlit object): A Streamlit object to display the detected video.
        image (numpy array): A numpy array representing the video frame.
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Ensure image is converted to BGR format (if needed)
    if cv2.cvtColor(image, cv2.COLOR_BGR2RGB).shape == image.shape:
        # No conversion needed (assuming input is BGR)
        pass
    else:
        # Convert to BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGB)

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

def detect_webcam(conf, model):
    """
    Performs real-time object detection using YOLOv8 on the webcam.

    Args:
        conf (float): Confidence threshold for object detection.
        model (YOLO): An instance of the `YOLOv8` class containing the YOLOv8 model.
    """

    try:
        # Capture video from webcam
        vid_cap = cv2.VideoCapture(0)  # 0 for default webcam

        # Create an empty Streamlit frame for displaying the video
        st_frame = st.empty()

        while True:
            success, image = vid_cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                # If a frame is not read successfully, continue to the next iteration
                continue

            # Perform object detection and display the frame
            _display_detected_frames(conf, model, st_frame, image)

            # Exit loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video capture and close all windows
        vid_cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        st.error("Error accessing webcam:", e)

# ... Rest of your Streamlit app code ...

# Main section for user interaction
st.title("Real-Time Object Detection")
confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Load the YOLOv8 model
model = load_model("/Users/syntichemusawu/Desktop/MidtermProject/best.pt")  # Replace with your model path

# Display webcam feed with object detection
st.header("Webcam Detection")
detect_webcam(confidence_threshold, model)