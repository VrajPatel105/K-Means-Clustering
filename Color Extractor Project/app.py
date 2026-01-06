# Vraj Patel - 6th January 2026
import streamlit as st
import pandas as pd
import numpy as np
from tempfile import NamedTemporaryFile
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


st.write("Please upload an image here")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png"])
if uploaded_file is not None:
   with NamedTemporaryFile(delete=False) as temp_file:
       temp_file.write(uploaded_file.getbuffer())
       temp_file_path = temp_file.name
    
image = plt.imread(temp_file_path)

st.image(temp_file_path)
st.write("Top 5 Major Colors")

X = image.reshape(-1,3)


kmeans = KMeans(n_clusters=5)
kmeans.fit(X)

from PIL import Image, ImageDraw
import numpy as np

def create_color_palette(dominant_colors, palette_size=(500, 100)):
    # Create an image to display the colors
    palette = Image.new("RGB", palette_size)
    draw = ImageDraw.Draw(palette)

    # Calculate the width of each color swatch
    swatch_width = palette_size[0] // len(dominant_colors)

    # Draw each color as a rectangle on the palette
    for i, color in enumerate(dominant_colors):
        draw.rectangle([i * swatch_width, 0, (i + 1) * swatch_width, palette_size[1]], fill=tuple(color))

    return palette

st.image(create_color_palette(kmeans.cluster_centers_.astype(int)))