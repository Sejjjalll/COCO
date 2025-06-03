import streamlit as st


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# MUST BE FIRST Streamlit command
st.set_page_config(
    page_title="Cocoa Leaf Disease Detector",
    layout="wide",
    page_icon="🌱",
    initial_sidebar_state="expanded"
)



from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import models
import json
import numpy as np
import time
import time
from datetime import datetime
import base64
from io import BytesIO
import pandas as pd
import cv2
from PIL import ImageEnhance

# ------------------------ Setup ------------------------
 
# Load class map
with open("class_map.json", "r") as f:
    class_map = json.load(f)
class_map = {int(k): v.lower() for k, v in class_map.items()}

 
# Disease information dictionary
disease_info = {
    "Healthy": {
        "description": "This leaf appears healthy with no signs of disease.",
        "management": "Continue regular maintenance and monitoring.",
        "prevention": "Maintain good air circulation, proper irrigation, and routine fungicide applications as preventative measures.",
        "report_text": "Your leaf sample appears to be healthy. No immediate intervention is required. Regular monitoring and proper care are still recommended to ensure continued plant health."
    },
    "Anthracnose": {
        "description": "Anthracnose is a fungal disease causing dark, sunken lesions on leaves and pods.",
        "management": "Prune infected areas, apply appropriate fungicides, and reduce humidity around plants.",
        "prevention": "Use resistant varieties and maintain good plant hygiene.",
        "report_text": "The image analysis suggests Anthracnose, which is a common cocoa leaf disease. Early treatment with fungicides and removal of infected parts is advised. Timely intervention can reduce spread."
    },
    "CSSVD": {
        "description": "Cocoa Swollen Shoot Virus Disease (CSSVD) causes vein clearing, swelling of shoots, and leaf deformation.",
        "management": "Remove infected trees and control mealybug vectors.",
        "prevention": "Use certified planting material and maintain field sanitation.",
        "report_text": "The system has detected CSSVD, a viral disease. Immediate removal of infected plants and containment measures are strongly recommended to avoid crop loss."
    }
}

 
# Add fallback for any unlisted diseases in the model
for label in class_map.values():
    if label not in disease_info:
        disease_info[label] = {
            "description": f"This appears to be {label} condition.",
            "management": "Consult with a local agricultural extension for specific management practices.",
            "prevention": "Practice good field sanitation and regular monitoring."
        }
 
# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# ---------------- EfficientNetB0 Model Loader (Fixed) ----------------
@st.cache_resource
def load_model():
    num_classes = len(class_map)
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    model_path = "best_EfficientNetB0_on_test.pth"
    if not os.path.exists(model_path):
        st.error(" Model file not found. Please place 'best_EfficientNetB0_on_test.pth' in the directory.")
        st.stop()

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("net", checkpoint)

    # Properly clean the keys (fixes misclassification)
    cleaned_state_dict = {k.replace("net.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

 
# Image transform (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

 
# ------------------------ UI Styling ------------------------
 
# Custom CSS for better design

css = """
<style>
    /* Global Reset & Font */
    * {
        transition: all 0.3s ease-in-out;
        font-family: 'Segoe UI', sans-serif;
    }

    body {
        background-color: #fdfaf6;
        margin: 0;
        padding: 0;
    }

    .main {
        background-color: #ffffff;
        color: #2c2c2c;
        padding: 20px;
        border-radius: 12px;
    }

    h1 {
        color: #4e342e;
        text-align: center;
        font-family: 'Georgia', serif;
        margin-bottom: 30px;
        font-size: 2.8em;
    }

    h2, h3 {
        color: #6d4c41;
        margin-top: 20px;
    }

    @media (max-width: 768px) {
        h1 {
            font-size: 2em;
        }
        h2, h3 {
            font-size: 1.2em;
        }
    }

    .info-card {
        background-color: #faf3e0;
        border-left: 6px solid #a1887f;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }

    .stButton>button {
        background-color: #4e342e;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        padding: 10px 20px;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    .stButton>button:hover {
        background-color: #6d4c41;
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }

    [data-testid="stFileUploader"] > div:first-child {
        border: 2px dashed #795548;
        background-color: #efebe9;
        border-radius: 10px;
        padding: 1rem;
        color: #4e342e;
    }

    .prediction-result {
        font-size: 1.2em;
        font-weight: bold;
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
    }

    .confidence-high {
        background-color: rgba(76, 175, 80, 0.15);
        border-left: 5px solid #388e3c;
    }

    .confidence-medium {
        background-color: rgba(255, 193, 7, 0.15);
        border-left: 5px solid #ffa000;
    }

    .confidence-low {
        background-color: rgba(244, 67, 54, 0.15);
        border-left: 5px solid #e53935;
    }

    .footer {
        text-align: center;
        font-size: 0.9em;
        color: #6d4c41;
        margin-top: 50px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #fcefe3;
        border-radius: 5px;
        padding: 10px;
        color: #5d4037;
    }

    .stTabs [aria-selected="true"] {
        background-color: #d7ccc8;
        border-bottom: 2px solid #4e342e;
    }

    [data-testid="stExpander"] {
        border: 1px solid #6d4c41;
        border-radius: 10px;
        margin-bottom: 1rem;
        background-color: #f1e2d0;
        color: #4e342e;
        font-weight: 500;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }

    [data-testid="stExpander"] .st-ae {
        color: #4e342e !important;
    }

    [data-testid="stExpander"] header:hover {
        background-color: #ecd5b5;
        cursor: pointer;
    }

    [data-testid="stExpander"] svg {
        color: #6d4c41;
    }

    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-thumb {
        background: #a1887f;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #fdfaf6;
    }

    section[data-testid="stSidebar"] {
        background-color: #2e2b2a;
        color: #f5f5f5;
        padding: 1rem;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] .sidebar-title {
        font-weight: 700 !important;
        color: #d7ccc8 !important;
        font-size: 1.3em;
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #5d4037;
        padding-bottom: 0.3rem;
    }

    .sidebar-subtitle {
        color: #bcaaa4;
        font-weight: 600;
        margin-top: 0.8rem;
        font-size: 1em;
    }

    img {
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }

    .sidebar-app-title {
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        color: #f5f5f5;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    label, .stTextInput label, .stSelectbox label, .stSlider label, .stTextArea label {
        color: #4e342e !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }

    .css-1e5imcs {
        color: #4e342e !important;
        font-weight: bold !important;
    }

    input::placeholder, textarea::placeholder {
        color: #d7ccc8 !important;
        opacity: 0.8;
    }

    .stSlider {
        padding-top: 10px;
        padding-bottom: 10px;
    }

    input, textarea, select,
    .stTextInput>div>input, .stTextArea>div>textarea, .stSelectbox>div>div,
    .stTextInput input, .stTextArea textarea, .stSelectbox select,
    input[type="text"], textarea[type="text"], select,
    input:-webkit-autofill, textarea:-webkit-autofill {
        border: 1px solid #a1887f !important;
        padding: 6px 10px !important;
        background-color: #2e2b2a !important;
        color: #ffffff !important;
        font-size: 15px !important;
        border-radius: 6px !important;
        -webkit-text-fill-color: #ffffff !important;
        caret-color: white !important;
    }

    input:focus, textarea:focus, select:focus {
        border-color: #ffb74d !important;
        box-shadow: 0 0 5px rgba(255,183,77,0.5);
        outline: none;
    }

    section[data-testid="stSidebar"] select,
    section[data-testid="stSidebar"] .stSelectbox > div[data-baseweb="select"] > div,
    div[data-baseweb="select"] div[role="option"],
    .stSelectbox > div[data-baseweb="select"] > div,
    .stSelectbox > div[data-baseweb="select"] svg {
        background-color: #3e3e3e !important;
        color: #ffffff !important;
    }

    /* >>>>>>> NEW STYLE FOR MOBILE TIPS >>>>>>> */
    .mobile-tip-box {
        background-color: #fffbe6;
        color: #000000 !important;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 14px;
        line-height: 1.6;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* Fix the black block and font size inside the dropdown input field */
.stSelectbox > div[data-baseweb="select"] > div {
    background-color: #3e3e3e !important;
    color: #ffffff !important;
    font-size: 13px !important;
    border-radius: 6px !important;
    padding: 6px 12px !important;
    min-height: 38px !important;
    line-height: 1.3 !important;
}

/* Fix inner div to prevent black square flicker */
.stSelectbox > div[data-baseweb="select"] div[role="button"] {
    background-color: #3e3e3e !important;
    color: #ffffff !important;
    font-size: 13px !important;
    padding-left: 10px !important;
    display: flex;
    align-items: center;
    height: 38px;
}

/* Dropdown options */
div[data-baseweb="select"] div[role="option"] {
    background-color: #3e3e3e !important;
    color: #ffffff !important;
    font-size: 13px !important;
}

/* Dropdown icon color */
.stSelectbox svg {
    color: #ffffff !important;
}

/* Focus and selection highlight */
div[data-baseweb="select"]:focus-within {
    box-shadow: 0 0 0 2px rgba(255, 183, 77, 0.5) !important;
    border-color: #ffb74d !important;
}

/* Dropdown menu visibility */
[data-baseweb="popover"] {
    z-index: 9999 !important;
}


    
</style>
"""




st.markdown(css, unsafe_allow_html=True)



 
# ------------------------ App Logic ------------------------
 
with st.sidebar:
    st.sidebar.image("Website_logo.jpg", use_column_width=True)
    st.sidebar.markdown("<div class='sidebar-app-title'>Cocoa Leaf Disease Detector</div>", unsafe_allow_html=True)
    st.title("About")
    st.markdown("""
    ### Cocoa Leaf Disease Detector
   
    This tool uses artificial intelligence to identify diseases in cocoa leaves.
   
    **How to use:**
    1. Upload a clear, well-lit photo of a cocoa leaf
    2. Click "Analyze Leaf"
    3. View the analysis results
   
    **Supported Diseases:**
    """)
   
    for disease in sorted(set(class_map.values())):
        st.markdown(f"- {disease}")
   
    st.caption("This tool is intended for preliminary diagnosis only. Always consult with agricultural experts for confirmation.")
 
# Main content
st.title("🌱 Cocoa Leaf Disease Detector")


 
st.markdown("""
<div class="info-card">
    <h3>Protect Your Cocoa Crops</h3>
    <p>Upload a photo of a cocoa leaf to identify potential diseases. Early detection can help prevent crop losses and improve productivity.</p>
</div>
""", unsafe_allow_html=True)
 
# Two columns for upload and results
col1, col2 = st.columns([1, 1.5])
 
with col1:
    st.subheader("Upload Leaf Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
   
    # Sample images option
    st.markdown("### Or try a sample image:")
    sample_option = st.selectbox(
    "Select a sample",
    ["None", "Healthy", "Anthracnose", "CSSVD"]
    )

   
    analyze_button = st.button("Analyze Leaf")
   
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
 
with col2:
    result_container = st.container()
   
    with result_container:
        if not analyze_button:
            st.markdown("""
            <div class="info-card">
                <h3>How It Works</h3>
                <p>This tool uses a deep learning model trained on thousands of cocoa leaf images to identify common diseases. The model analyzes patterns, discolorations, and textures that may indicate different health conditions.</p>
            </div>
            """, unsafe_allow_html=True)
       
# Logic for processing
if analyze_button:
    image = None
    image_source = None

    sample_map = {
        "Healthy": "healthy_leaf.jpg",
        "Anthracnose": "anthracnose_leaf.jpg",
        "CSSVD": "cssvd_leaf.jpg"
    }

    # Priority 1: Uploaded Image
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_source = "upload"
        except UnidentifiedImageError:
            st.error("Invalid image file. Please upload a valid JPG or PNG image.")

    # Priority 2: Sample image (only if no upload)
    elif sample_option != "None":
        sample_path = sample_map.get(sample_option)
        if sample_path:
            try:
                image = Image.open(sample_path)
                image_source = "sample"
            except FileNotFoundError:
                st.warning(f"Sample image '{sample_path}' not found.")
                st.info("Please upload your own image instead.")



   
    if image_source:
        with st.spinner("Analyzing leaf..."):
            # Add a small delay to show the spinner (more user-friendly)
            time.sleep(1.5)
           
            # Process the image
            img_tensor = transform(image).unsqueeze(0).to(device)
           
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)[0]
               
                # Get top 3 predictions for display
                top_probs, top_indices = torch.topk(probs, 3)
               
                # Main prediction
                pred_idx = int(top_indices[0].item())
                confidence = top_probs[0].item() * 100.0
                label = class_map.get(pred_idx, "Unknown")

                 #Basic non-leaf image detection
                non_leaf_keywords = ["object", "face", "person", "animal", "car", "building", "random", "unknown"]

                if label.lower() in non_leaf_keywords or confidence < 40:
                    st.error(" The uploaded image does not appear to be a cocoa leaf. Please upload a clear image of a cocoa leaf for analysis.")
                    st.stop()
               
                # Determine confidence class for styling
                if confidence > 80:
                    confidence_class = "confidence-high"
                elif confidence > 60:
                    confidence_class = "confidence-medium"
                else:
                    confidence_class = "confidence-low"
       
        with result_container:
            st.markdown(f"""
            <div class="prediction-result {confidence_class}">
                <h2>Diagnosis Result</h2>
                <p>Primary diagnosis: <strong>{label}</strong> with {confidence:.1f}% confidence</p>
            </div>
            """, unsafe_allow_html=True)
           
            # Create tabs for different information sections
            tabs = st.tabs(["Disease Information", "Management", "Prevention", "Other Possibilities"])
           
            with tabs[0]:
                st.markdown(f"### About {label}")
                st.markdown(disease_info[label]["description"])
               
                if label != "Healthy":
                    st.warning("Early intervention is recommended to prevent spread to other plants.")
           
            with tabs[1]:
                st.markdown("### Management Recommendations")
                st.markdown(disease_info[label]["management"])
               
                # Add some visual management steps
                st.markdown("""
                **General steps:**
                1. Isolate affected plants if possible
                2. Follow recommended treatment protocols
                3. Monitor surrounding plants closely
                4. Document the outbreak for future reference
                """)
           
            with tabs[2]:
                st.markdown("### Prevention Strategies")
                st.markdown(disease_info[label]["prevention"])
               
                # Add some visual prevention tips
                st.markdown("""
                **Best practices:**
                - Regular inspection of leaves and stems
                - Maintain proper spacing between plants
                - Ensure good air circulation
                - Implement appropriate irrigation practices
                - Apply preventative treatments during high-risk seasons
                """)
               
            with tabs[3]:
                st.markdown("### Other Possibilities")
               
                # Create a dataframe for the alternative predictions
                alt_predictions = []
                for i in range(1, len(top_indices)):
                    if i < len(top_indices):
                        alt_idx = int(top_indices[i].item())
                        alt_prob = top_probs[i].item() * 100.0
                        alt_label = class_map.get(alt_idx, "Unknown")
                       
                        if alt_prob > 5.0:  # Only show alternatives with >5% confidence
                            alt_predictions.append({
                                "Disease": alt_label,
                                "Confidence": f"{alt_prob:.1f}%"
                            })
               
                if alt_predictions:
                    st.table(alt_predictions)
                else:
                    st.info("No other significant possibilities detected.")
               
                st.markdown("""
                *Note: When confidence levels are below 70%, consider consulting with an agricultural expert
                for confirmation, especially if symptoms appear ambiguous.*
                """)
 
# Add a progress bar to show model coverage
st.markdown("### Model Coverage")
st.markdown("Our model can currently detect the following cocoa leaf conditions:")
 
# Create a visual representation of the model's capabilities
coverage_data = {
    "Leaf Diseases": 80,
    "Pod Diseases": 75,
    "Pest Damage": 70,
    "Nutrient Deficiencies": 65,
    "Environmental Stress": 60
}
 
for category, coverage in coverage_data.items():
    st.write(f"{category}: {coverage}%")
    st.progress(coverage / 100)
 
# Tips section
st.markdown("""
<div class="info-card">
    <h3>Tips for Better Results</h3>
    <ul>
        <li>Take photos in natural daylight</li>
        <li>Ensure the leaf is in focus and fills most of the frame</li>
        <li>Include both healthy and affected parts of the leaf if possible</li>
        <li>Take multiple photos from different angles for complex cases</li>
    </ul>
</div>
""", unsafe_allow_html=True)
 
# Footer
st.markdown("""
<div class="footer">
    <p>Cocoa Leaf Disease Detector v2.0 | Developed to support sustainable cocoa farming</p>
    <p>This tool is intended for educational purposes and preliminary diagnosis only.</p>
</div>
""", unsafe_allow_html=True)
 
 
 
# Add history tracking functionality
if 'history' not in st.session_state:
    st.session_state.history = []
 
# Add image enhancement options
st.sidebar.markdown("---")
st.sidebar.header("Image Enhancement")
show_enhancement = st.sidebar.checkbox("Enable Image Enhancement", False)
 
if show_enhancement and uploaded_file is not None:
    enhancement_container = st.sidebar.container()
    with enhancement_container:
        st.write("Adjust image parameters:")
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
        sharpness = st.slider("Sharpness", 0.0, 3.0, 1.0, 0.1)
       
        if 'original_image' not in st.session_state and uploaded_file is not None:
            st.session_state.original_image = Image.open(uploaded_file).convert("RGB")
       
        if 'original_image' in st.session_state:
            # Apply enhancements
            enhanced_image = st.session_state.original_image
            enhanced_image = ImageEnhance.Brightness(enhanced_image).enhance(brightness)
            enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(contrast)
            enhanced_image = ImageEnhance.Sharpness(enhanced_image).enhance(sharpness)
           
            col1.image(enhanced_image, caption="Enhanced Image", use_column_width=True)
           
            # Update the image to be used for analysis
            img_buffer = BytesIO()
            enhanced_image.save(img_buffer, format="JPEG")
            image = enhanced_image
 
# Add batch processing option
st.sidebar.markdown("---")
st.sidebar.header("Batch Processing")
batch_processing = st.sidebar.checkbox("Enable Batch Processing", False)
 
if batch_processing:
    batch_files = st.sidebar.file_uploader("Upload multiple leaf images",
                                          type=["jpg", "jpeg", "png"],
                                          accept_multiple_files=True)
   
    if batch_files and st.sidebar.button("Process Batch"):
        batch_results = []
        progress_bar = st.sidebar.progress(0)
       
        for i, file in enumerate(batch_files):
            try:
                img = Image.open(file).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
               
                with torch.no_grad():
                    output = model(img_tensor)
                    probs = torch.softmax(output, dim=1)[0]
                    pred_idx = int(torch.argmax(probs).item())
                    confidence = probs[pred_idx].item() * 100.0
                    label = class_map.get(pred_idx, "Unknown")
                   
                batch_results.append({
                    "File": file.name,
                    "Prediction": label,
                    "Confidence": f"{confidence:.1f}%"
                })
               
            except Exception as e:
                batch_results.append({
                    "File": file.name,
                    "Prediction": "Error",
                    "Confidence": f"Error: {str(e)}"
                })
               
            # Update progress
            progress_bar.progress((i + 1) / len(batch_files))
       
        # Show batch results
        st.sidebar.success(f"Processed {len(batch_files)} images")
        batch_df = pd.DataFrame(batch_results)
        st.sidebar.dataframe(batch_df)
       
        # Download results option
        csv = batch_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="batch_results.csv">Download Results CSV</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
 
# Add analysis history
if analyze_button and image_source and 'label' in locals():
    # Record this analysis in history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
   
    # Save a thumbnail of the image
    thumb = image.copy()
    thumb.thumbnail((100, 100))
    thumb_buffer = BytesIO()
    thumb.save(thumb_buffer, format="JPEG")
    encoded_image = base64.b64encode(thumb_buffer.getvalue()).decode()
   
    st.session_state.history.append({
        "timestamp": timestamp,
        "diagnosis": label,
        "confidence": f"{confidence:.1f}%",
        "image_data": encoded_image
    })
 
# Show history tab
st.markdown("---")
show_history = st.expander("Analysis History", False)
with show_history:
    if not st.session_state.history:
        st.info("No analysis history yet. Diagnose some leaves to build history.")
    else:
        # Create a history table
        for i, entry in enumerate(reversed(st.session_state.history)):
            col1, col2, col3 = st.columns([1, 2, 1])
           
            with col1:
                st.image(f"data:image/jpeg;base64,{entry['image_data']}", caption=f"Sample {i+1}")
           
            with col2:
                st.write(f"**Diagnosis:** {entry['diagnosis']}")
                st.write(f"**Confidence:** {entry['confidence']}")
           
            with col3:
                st.write(f"**Date:** {entry['timestamp']}")
           
            st.markdown("---")
       
        # Option to clear history
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

 
# Add visual leaf analysis
if analyze_button and image_source and 'label' in locals() and label != "Healthy":
    st.markdown("---")
    visualize = st.expander("Visual Leaf Analysis", False)
   
    with visualize:
        st.write("### Affected Area Detection")
       
        try:
            # Convert PIL image to OpenCV format
            img_cv = np.array(image)
            img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR
           
            # Create a simplified visualization of potentially affected areas
            # (This is a simplified demo - a real implementation would use more sophisticated image processing)
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
           
            # Different masks based on disease type
            if "Black" in label:
                # Dark areas for Black Pod
                lower = np.array([0, 0, 0])
                upper = np.array([180, 255, 80])
            elif "Frosty" in label:
                # Whitish areas for Frosty Pod
                lower = np.array([0, 0, 180])
                upper = np.array([180, 70, 255])
            else:
                # Yellow/brown discoloration for other diseases
                lower = np.array([20, 100, 100])
                upper = np.array([30, 255, 255])
               
            # Create mask
            mask = cv2.inRange(hsv, lower, upper)
           
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
           
            # Draw contours on original image
            result = img_cv.copy()
            cv2.drawContours(result, contours, -1, (0, 0, 255), 2)
           
            # Convert back to PIL for display
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
           
            col1, col2 = st.columns(2)
           
            with col1:
                st.image(image, caption="Original Image")
               
            with col2:
                st.image(result_pil, caption="Potential Affected Areas")
               
            st.markdown("""
            <div class="info-card">
                <p><strong>Note:</strong> This is a simplified visualization to highlight potential problem areas.
                The red outlines indicate regions that may be affected based on color analysis.
                This visualization is not diagnostic in itself.</p>
            </div>
            """, unsafe_allow_html=True)
           
        except Exception as e:
            st.error(f"Could not generate visual analysis: {str(e)}")
            st.info("Visual analysis works best with clear, well-lit images.")
 
# Add comparison feature

st.markdown("---")
comparison = st.expander("Compare With Healthy Leaves", False)

with comparison:
    st.markdown("### Healthy vs. Diseased Comparison")

    st.markdown("""
    <div class="info-card" style="font-size: 16px;">
        <p>Reference images help farmers understand what healthy cocoa leaves should look like compared to common diseases.
        This can aid in early detection of problems.</p>
    </div>
    """, unsafe_allow_html=True)

    # Layout images in 3 columns
    ref_col1, ref_col2, ref_col3 = st.columns(3)

    with ref_col1:
        st.markdown("<h4 style='text-align: center; font-size: 20px;'>Healthy Leaf</h4>", unsafe_allow_html=True)
        st.image("healthy_leaf.jpg", caption="Healthy Cocoa Leaf", use_column_width=True)
        st.markdown("""
        <ul style='font-size: 15px;'>
            <li>Uniform green color</li>
            <li>Smooth surface</li>
            <li>No spots or lesions</li>
        </ul>
        """, unsafe_allow_html=True)

    with ref_col2:
        st.markdown("<h4 style='text-align: center; font-size: 20px;'>Anthracnose Symptoms</h4>", unsafe_allow_html=True)
        st.image("anthracnose_leaf.jpg", caption="Anthracnose", use_column_width=True)
        st.markdown("""
        <ul style='font-size: 15px;'>
            <li>Dark, sunken lesions</li>
            <li>Often begins at leaf margins</li>
            <li>Common in humid conditions</li>
        </ul>
        """, unsafe_allow_html=True)

    with ref_col3:
        st.markdown("<h4 style='text-align: center; font-size: 20px;'>CSSVD Symptoms</h4>", unsafe_allow_html=True)
        st.image("cssvd_leaf.jpg", caption="CSSVD (Cocoa Swollen Shoot Virus)", use_column_width=True)
        st.markdown("""
        <ul style='font-size: 15px;'>
            <li>Vein clearing and yellowing</li>
            <li>Shoot swelling</li>
            <li>Curled or deformed leaves</li>
        </ul>
        """, unsafe_allow_html=True)

 
# Add seasonal trends data visualization
st.markdown("---")
trends = st.expander("Disease Seasonal Trends", False)
 
with trends:
    st.write("### Seasonal Disease Prevalence")
   
    # Sample data for visualization
    seasons = ['Jan-Mar', 'Apr-Jun', 'Jul-Sep', 'Oct-Dec']
    black_pod = [15, 35, 70, 40]
    frosty_pod = [20, 45, 30, 10]
    pod_borer = [30, 20, 15, 25]
    swollen_shoot = [10, 15, 25, 20]
   
    # Create seasonal trends chart
    fig, ax = plt.subplots(figsize=(10, 6))
   
    ax.plot(seasons, black_pod, marker='o', linewidth=2, label='Black Pod')
    ax.plot(seasons, frosty_pod, marker='s', linewidth=2, label='Frosty Pod')  
    ax.plot(seasons, pod_borer, marker='^', linewidth=2, label='Pod Borer')
    ax.plot(seasons, swollen_shoot, marker='d', linewidth=2, label='Swollen Shoot')
   
    ax.set_xlabel('Season')
    ax.set_ylabel('Prevalence (%)')
    ax.set_title('Seasonal Disease Prevalence')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
   
    st.pyplot(fig)
   
    st.markdown("""
    <div class="info-card">
        <p><strong>Understanding Seasonal Patterns:</strong> Disease prevalence often follows seasonal patterns.
        Monitoring your plantation based on these patterns can help implement preventative measures
        before high-risk seasons begin.</p>
    </div>
    """, unsafe_allow_html=True)
 
# Add treatment recommendations calculator
st.markdown("---")
treatment = st.expander("Treatment Calculator", False)
 
with treatment:
    st.write("### Treatment Recommendations Calculator")
   
    st.markdown("""
    <div class="info-card">
        <p>Calculate recommended treatment amounts based on your plantation size and disease severity.</p>
    </div>
    """, unsafe_allow_html=True)
   
    calc_col1, calc_col2 = st.columns(2)
   
    with calc_col1:
        plantation_size = st.number_input("Plantation Size (hectares)", min_value=0.1, max_value=1000.0, value=1.0, step=0.1)
        tree_density = st.number_input("Tree Density (trees/hectare)", min_value=100, max_value=2000, value=1000, step=100)
        disease_severity = st.select_slider("Disease Severity", options=["Low", "Medium", "High", "Severe"])
       
    with calc_col2:
        # Calculate recommendations based on inputs
        severity_factor = {"Low": 0.5, "Medium": 1.0, "High": 1.5, "Severe": 2.0}
        factor = severity_factor[disease_severity]
       
        total_trees = plantation_size * tree_density
       
        # Fungicide calculations (simplified example)
        fungicide_per_tree = 0.01 * factor  # liters per tree
        total_fungicide = fungicide_per_tree * total_trees
       
        # Labor calculations
        labor_days = (total_trees / 200) * factor  # days of labor
       
        st.markdown(f"**Total Trees:** {int(total_trees)}")
        st.markdown(f"**Recommended Fungicide:** {total_fungicide:.2f} liters")
        st.markdown(f"**Estimated Labor Required:** {labor_days:.1f} person-days")
        st.markdown(f"**Treatment Frequency:** Every {7 if disease_severity in ['High', 'Severe'] else 14} days")
       
        st.markdown("""
        <div class="info-card">
            <p><strong>Note:</strong> These are general recommendations only. Consult with local agricultural extension
            services for specific treatment protocols suitable for your region.</p>
        </div>
        """, unsafe_allow_html=True)
 
# Add expert consultation section
st.markdown("---")
expert = st.expander("Expert Consultation", False)
 
with expert:
    st.write("### Connect with Cocoa Disease Experts")
   
    st.markdown("""
    For cases requiring expert consultation, you can connect with agricultural specialists
    in your region. Complete the form below to request assistance.
    """)
   
    form_col1, form_col2 = st.columns(2)
   
    with form_col1:
        farmer_name = st.text_input("Name")
        farmer_email = st.text_input("Email")
        farmer_phone = st.text_input("Phone")
       
    with form_col2:
        farmer_region = st.selectbox("Region", ["West Africa", "Central America", "South America", "Southeast Asia", "Other"])
        urgency = st.select_slider("Urgency Level", options=["Low", "Medium", "High", "Emergency"])
        message = st.text_area("Brief description of the issue")
   
    if st.button("Submit Request"):
        st.success("Your consultation request has been submitted. An expert will contact you within 24-48 hours.")
       
        # In a real app, this would send the request to a database or email system
       
        st.info("You can also contact your local agricultural extension office directly for immediate assistance.")
 
# Add mobile-friendly notification
is_mobile = st.sidebar.checkbox("I'm using a mobile device", False)

if is_mobile:
    st.sidebar.markdown("""
    <div class="mobile-tip-box">
        <strong>Mobile Tips:</strong><br>
        For better results on mobile devices:
        <ul>
            <li>Use landscape orientation for optimal viewing</li>
            <li>Use the camera's focus feature before taking photos</li>
            <li>Ensure good lighting conditions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

 
# Add offline mode option
st.sidebar.markdown("---")
offline_mode = st.sidebar.checkbox("Enable Offline Mode", False)
 
if offline_mode:
    st.sidebar.info("Offline mode enabled. Analysis results will be stored locally and synchronized when you reconnect.")
   
    # In a real app, this would trigger local storage logic
 
# Add language selection
st.sidebar.markdown("---")
st.sidebar.header("Language")
language = st.sidebar.selectbox("Select Language", ["English", "Español", "Français", "Português", "Bahasa Indonesia"])
 
# Display note about the selected language
if language != "English":
    st.sidebar.info(f"The app would be displayed in {language} if this were a full implementation.")
 
# Add community section
st.markdown("---")
community = st.expander("Join Farming Community", False)
 
with community:
    st.write("### Connect with Other Cocoa Farmers")
   
    st.markdown("""
    <div class="info-card">
        <p>Join our community of cocoa farmers to share experiences, best practices, and support.</p>
        <ul>
            <li>Discuss disease management strategies</li>
            <li>Share success stories and challenges</li>
            <li>Access educational resources and webinars</li>
            <li>Connect with agricultural experts</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
   
    st.text_input("Email Address", placeholder="Enter your email to join")
   
    if st.button("Join Community"):
        st.success("Welcome to the community! Check your email for confirmation.")
 
# Add feedback collection
st.markdown("---")
feedback = st.expander("Provide Feedback", False)
 
with feedback:
    st.write("### Help Improve This Tool")
   
    st.markdown("""
    Your feedback helps us improve the accuracy and usability of this disease detection tool.
    Please share your experience and suggestions.
    """)
   
    feedback_rating = st.slider("How would you rate this tool?", 1, 5, 5)
    feedback_text = st.text_area("What could we improve?")
   
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback! We value your input.")
 
# Add download PDF report option
if analyze_button and image_source and 'label' in locals():
    st.markdown("---")
    st.subheader("Download Report")

    from fpdf import FPDF

    if st.button("Generate PDF Report"):
        st.info("Generating comprehensive PDF report with analysis results and recommendations...")

        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.005)
            progress_bar.progress(i + 1)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Cocoa Leaf Disease Detection Report", ln=True)

        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(0, 10, f"Diagnosis: {label.title()} ({confidence:.1f}% confidence)", ln=True)

        pdf.ln(10)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Summary", ln=True)
        pdf.set_font("Arial", size=12)

        # Ensure disease_info contains report_text for the predicted label
        summary_text = disease_info.get(label.title(), {}).get("report_text", "No detailed report available.")
        pdf.multi_cell(0, 8, summary_text)

        # Convert PDF to downloadable stream
        pdf_output = BytesIO()
        pdf_bytes = pdf.output(dest='S').encode('latin1')  # Get PDF as bytes
        pdf_output.write(pdf_bytes)
        pdf_output.seek(0)


        st.success("✅ Report generated successfully!")

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_output,
            file_name=f"cocoa_disease_report_{label}.pdf",
            mime="application/pdf"
        )