import os
import sys
# Fix for Streamlit Cloud deployment: Change working directory to the app's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');
    
    .stApp {
        background: radial-gradient(circle at top left, #0f172a, #000000);
        color: #e2e8f0;
        font-family: 'Roboto', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #38bdf8;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #38bdf8 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.3);
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.6);
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
    }
    
    .css-1d391kg {
        background-color: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    input, select, textarea {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        color: white !important;
        border-radius: 6px !important;
    }
    
    .premium-footer {
        text-align: center;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid rgba(255,255,255,0.1);
        font-size: 0.8rem;
        color: #64748b;
        letter-spacing: 1px;
    }
</style>
''', unsafe_allow_html=True)

import joblib
import numpy as np
from PIL import Image, ImageOps, ImageFilter

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model. Make sure 'model.pkl' exists. {e}")
    st.stop()

st.title("Digit Recognition App")
st.write("Upload an image of a digit to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=200)
        st.write("")
        st.write("Classifying...")

        # --- Preprocessing Pipeline ---
        # Goal: Match the format of sklearn load_digits (8x8, values 0-16 with GRADIENTS)
        
        # 1. Convert to grayscale
        img_gray = ImageOps.grayscale(image)
        
        # 2. Invert if background is light (make digit white on black)
        img_array = np.array(img_gray)
        if img_array.mean() > 128:
            img_gray = ImageOps.invert(img_gray)
            
        # 3. Find bounding box and crop (with some padding)
        bbox = img_gray.getbbox()
        if bbox:
            # Add small padding around the digit
            x1, y1, x2, y2 = bbox
            pad = int(max((x2-x1), (y2-y1)) * 0.1)  # 10% padding
            w, h = img_gray.size
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            img_cropped = img_gray.crop((x1, y1, x2, y2))
        else:
            img_cropped = img_gray
            
        # 4. Make it square by adding black padding
        w_crop, h_crop = img_cropped.size
        max_dim = max(w_crop, h_crop)
        img_square = Image.new('L', (max_dim, max_dim), color=0)
        paste_x = (max_dim - w_crop) // 2
        paste_y = (max_dim - h_crop) // 2
        img_square.paste(img_cropped, (paste_x, paste_y))
        
        # 5. Apply Gaussian blur to create smooth gradients (like training data)
        # This is KEY - the training data has anti-aliased digits, not sharp edges
        img_blurred = img_square.filter(ImageFilter.GaussianBlur(radius=1))
        
        # 6. Resize to 8x8 using high-quality downsampling
        img_final_pil = img_blurred.resize((8, 8), Image.Resampling.LANCZOS)
        
        # 7. Convert to numpy and scale to 0-16
        img_array = np.array(img_final_pil, dtype=np.float32)
        
        # Normalize to use full 0-16 range
        min_val = img_array.min()
        max_val = img_array.max()
        if max_val > min_val:
            img_scaled = (img_array - min_val) / (max_val - min_val) * 16.0
        else:
            img_scaled = np.zeros((8, 8))
            
        # 8. Flatten for prediction
        img_flat = img_scaled.reshape(1, -1)

        # Predict
        prediction = model.predict(img_flat)

        st.success(f"Predicted Digit: **{prediction[0]}**")
        
        # Visual Debug
        col1, col2 = st.columns(2)
        with col1:
            st.write("Processed (8x8):")
            img_display = img_final_pil.resize((128, 128), Image.Resampling.NEAREST)
            st.image(img_display, width=128)
        with col2:
            st.write("After Crop & Blur:")
            st.image(img_blurred.resize((128, 128), Image.Resampling.NEAREST), width=128)
        
        # Show raw values
        st.write("Raw Values (0-16) - Note: should have GRADIENTS, not just 0 and 16:")
        st.text(np.round(img_scaled, 1))

    except Exception as e:
        st.error(f"Error processing image: {e}")


st.markdown('<div class="premium-footer">Engineered by Partha Sarathi R</div>', unsafe_allow_html=True)
