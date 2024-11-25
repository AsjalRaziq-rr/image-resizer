import streamlit as st
import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n.pt')  # Ensure you have the YOLO model file in the working directory

# Streamlit app header
st.title("Image Resizer with Object Detection")
st.write("Upload an image to detect objects and resize it for social media.")

# File upload section
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Process the uploaded file
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

        # Check if objects are detected
        if results[0].boxes:
            # Extract bounding box for the first detected object
            x1, y1, x2, y2 = results[0].boxes[0].xyxy[0].tolist()

            # Crop the image to the bounding box region
            focal_area = image.crop((int(x1), int(y1), int(x2), int(y2)))

            # Resize the cropped image to platform requirements
            platform_size = (1080, 1080)  # Example size for Instagram
            resized_image = focal_area.resize(platform_size)

            # Save the resized image
            resized_file_name = f"resized_{uploaded_file.name}"
            resized_image.save(resized_file_name)

            # Display the resized image
            st.image(resized_image, caption="Resized Image", use_column_width=True)

            # Provide download link
            with open(resized_file_name, "rb") as file:
                st.download_button(
                    label="Download Resized Image",
                    data=file,
                    file_name=resized_file_name,
                    mime="image/png",
                )

            # Clean up temporary files
            os.remove(temp_file_path)
            os.remove(resized_file_name)
        else:
            st.warning("No objects detected in the uploaded image.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
