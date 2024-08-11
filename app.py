# streamlit_app.py
import streamlit as st
import pickle
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import albumentations as A
from albumentations import Compose, Resize, Normalize, HorizontalFlip
from albumentations.pytorch import ToTensorV2

st.markdown(
    """
    <style>
    .stApp {
        background-color: #5E9FE0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load your pre-trained U-net model from a .pkl file with CPU mapping
model = torch.load('entire_model.pkl', 
                map_location=torch.device('cpu'))
model.eval()

# Define a function to preprocess the image
def preprocess_image(image):
    # st.write('Inside preprocessing function ..')
    transform = Compose([
        Resize(256, 256),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    transformed = transform(image=image)
    return transformed['image'].unsqueeze(0)

# Define a function to perform segmentation
def segment_image(image):
    # st.write('About to Preprocess Image ..')
    input_tensor = preprocess_image(image)
    # st.write('Transformations Done :)')    
    with torch.no_grad():
        output = model(input_tensor)
    return output.squeeze().numpy()


# Streamlit interface
st.title("U-net Image Segmentation")
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
# Read the image using PIL and convert it to a NumPy array
    image = Image.open(uploaded_file)

    st.write('Name of the image file: ', uploaded_file.name)

    # Ensure the image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_np = np.array(image)
    # scaling the image
    # image_np = image_np / 255.0

    # st.write('Image Array Size: ', image_np.shape)
    # Display the original image
    # st.image(image_np, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Segmenting...")
    segmented_image = segment_image(image=image_np)
    scaled_segmented_image = np.clip(segmented_image, 0, 1)
    # predicted_mask = st.image(scaled_segmented_image, caption='Segmented    Image', use_column_width=True)


    if uploaded_file is not None and scaled_segmented_image is not None:
        # Read the images using PIL
        image1 = Image.open(uploaded_file)
        image1 = image.resize((256, 256))
        image2 = scaled_segmented_image

        # Create two columns
        col1, col2 = st.columns(2)

        # Display the first image in the first column
        with col1:
            st.image(image1, caption='Original Image', use_column_width=True)
        # Display the second image in the second column
        with col2:
            st.image(image2, caption='Segmented Image', use_column_width=True)