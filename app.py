import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n.pt')  # Ensure you have the YOLO model file in the working directory

# Streamlit app header
st.title("Image Resizer with Object Detection")
st.write("Upload an image, select an object, and resize it as per your preferences.")

# File upload section
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Save uploaded file to a temporary location
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Open the image with PIL
        image = Image.open(temp_file_path)

        # Perform object detection
        results = model(temp_file_path)
        detected_objects = results[0].boxes  # Get detected objects

        # Check if objects are detected
        if detected_objects:
            # Display detected objects for selection
            st.write("Detected objects:")
            object_options = [
                f"Object {i + 1} (Class: {int(box.cls[0])}, Confidence: {box.conf[0]:.2f})"
                for i, box in enumerate(detected_objects)
            ]
            selected_object = st.selectbox("Select an object to resize:", object_options)
            selected_index = object_options.index(selected_object)

            # Extract bounding box for the selected object
            x1, y1, x2, y2 = detected_objects[selected_index].xyxy[0].tolist()

            # Crop the image to the bounding box region
            focal_area = image.crop((int(x1), int(y1), int(x2), int(y2)))

            # User input for resize dimensions
            st.write("Customize resize dimensions:")
            custom_width = st.number_input("Width (px)", min_value=1, value=1080)
            custom_height = st.number_input("Height (px)", min_value=1, value=1080)
            platform_size = (custom_width, custom_height)

            # Resize the cropped image
            resized_image = focal_area.resize(platform_size)

            # User input for download format
            st.write("Select download format:")
            download_format = st.radio("Format", ["PNG", "JPG"])

            # Save the resized image
            resized_file_name = f"resized_{uploaded_file.name.split('.')[0]}.{download_format.lower()}"
            resized_image.save(resized_file_name, format=download_format)

            # Display the resized image
            st.image(resized_image, caption="Resized Image", use_column_width=True)

            # Provide download link
            with open(resized_file_name, "rb") as file:
                st.download_button(
                    label="Download Resized Image",
                    data=file,
                    file_name=resized_file_name,
                    mime=f"image/{download_format.lower()}",
                )

            # Clean up temporary files
            os.remove(temp_file_path)
            os.remove(resized_file_name)
        else:
            st.warning("No objects detected in the uploaded image.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
