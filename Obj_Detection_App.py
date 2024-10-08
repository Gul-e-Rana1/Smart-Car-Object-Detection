import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from urllib.request import urlopen
from object_detection.utils import label_map_util, visualization_utils as viz_utils

# Load TensorFlow Hub Model
model_handle = 'https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1'
hub_model = hub.load(model_handle)

# Load label map
PATH_TO_LABELS = './models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Function to load image into numpy array
def load_image_into_numpy_array(image):
    image = np.array(image)
    return image

# Streamlit Interface
st.title("Object Detection with TensorFlow Hub")

# Upload Image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert image to numpy array
    image_np = load_image_into_numpy_array(image)

    # Prepare image for model prediction
    image_np = image_np[tf.newaxis, ...]

    # Model Prediction
    results = hub_model(image_np)
    result = {key: value.numpy() for key, value in results.items()}

    # Visualization
    image_np_with_detections = image_np.copy()
    label_id_offset = 0

    # Extract keypoints and scores if available
    keypoints, keypoint_scores = None, None
    if 'detection_keypoints' in result:
        keypoints = result['detection_keypoints'][0]
        keypoint_scores = result['detection_keypoint_scores'][0]

    # Visualize detections on the image
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections[0],
        result['detection_boxes'][0],
        (result['detection_classes'][0] + label_id_offset).astype(int),
        result['detection_scores'][0],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores
    )

    # Display the image with detections
    st.image(image_np_with_detections[0], caption='Detected Objects', use_column_width=True)