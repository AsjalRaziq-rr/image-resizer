import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n.pt')

# Streamlit UI
st.title("Image Resizer & Focal Area Detector")
st.write("Upload an image, and this app will detect objects and resize the most prominent focal area.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform object detection using YOLO
    results = model(uploaded_file.name)

    # Check if any objects were detected
    if results[0].boxes:
        # Extract bounding box coordinates of the first detected object
        x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0].tolist())

        # Crop and resize the focal area
        focal_area = image.crop((x1, y1, x2, y2))
        resized_image = focal_area.resize((1080, 1080))  # Resize to Instagram-friendly size

        # Display the resized focal area
        st.image(resized_image, caption="Resized Focal Area", use_column_width=True)

        # Save the resized image for download
        resized_image.save("resized_image.png")
        with open("resized_image.png", "rb") as file:
            st.download_button(
                label="Download Resized Image",
                data=file,
                file_name="resized_image.png",
                mime="image/png"
            )
    else:
        st.warning("No objects detected in the image. Please try another image.")
