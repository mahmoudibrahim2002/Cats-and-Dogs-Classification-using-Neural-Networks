import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.let_it_rain import rain
import base64
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="🐾 Cats & Dogs Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

def set_dark_theme():
    # Set plotly dark theme
    plt.style.use('dark_background')
    
    # Custom dark theme for plotly charts
    plotly_template = {
        "layout": {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"color": "#e0e0e0"},
            "xaxis": {
                "gridcolor": "#3a3d48",
                "linecolor": "#3a3d48",
                "zerolinecolor": "#3a3d48"
            },
            "yaxis": {
                "gridcolor": "#3a3d48",
                "linecolor": "#3a3d48",
                "zerolinecolor": "#3a3d48"
            },
            "colorway": ["#6b8cff", "#ff6b6b", "#5cb85c", "#ffc107", "#a162e8", "#28a745"]
        }
    }
    import plotly.io as pio
    pio.templates["custom_dark"] = plotly_template
    pio.templates.default = "custom_dark"


set_dark_theme()

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Load the model
@st.cache_resource
def load_mobilenet_model():
    try:
        model = load_model('MobileNet_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_mobilenet_model()

# Image preprocessing
def preprocess_image(image, target_size=(96, 96)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Prediction function
def predict_image(image, model):
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    classes = {0: 'cat', 1: 'dog'}
    return classes[class_idx], confidence, prediction[0]

# Animated header
def animated_header():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px; animation: fadeIn 1s;">
        <h1 style="font-size: 2.8rem; 
                   background: linear-gradient(to right, #6b8cff, #ff6b6b);
                   -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent;
                   font-weight: 700;
                   margin-bottom: 0.5rem;">
            🐶 Cats and Dogs Classifier 🐱
        </h1>
        <p style="font-size: 1.1rem; color: var(--text-secondary);">
            Cats and Dogs Classification using Neural Networks
        </p>
    </div>
    """, unsafe_allow_html=True)

# Performance chart
def performance_chart():
    # Detailed performance data with training speed metrics
    data = {
        "Model": ["FFNN (Base)", "FFNN (Optimized)", 
                 "CNN (Base)", "CNN (Optimized)",
                 "LSTM (Base)", "LSTM (Optimized)",
                 "CNN-RNN (Base)", "CNN-RNN (Optimized)",
                 "MobileNetV2"],
        "Train Accuracy": [65.33, 96.44, 86.92, 98.85, 98.18, 94.99, 97.60, 98.90, 99.94],
        "Val Accuracy": [56.33, 85.63, 73.83, 93.67, 84.70, 87.06, 90.02, 93.48, 99.07],
        "Training Speed": ["⚡⚡⚡⚡", "⚡⚡⚡", "⚡⚡⚡", "⚡⚡", "⚡", "⚡", "⚡⚡", "⚡", "⚡⚡⚡ (after init)"],
        "Params (M)": [2.1, 3.5, 1.8, 2.4, 3.2, 3.8, 2.9, 3.1, 3.5],
        "Color": ["#FF6B6B", "#FF9E7D", "#FFD166", "#FFE66D", "#06D6A0", "#48BFE3", "#5390D9", "#5E60CE", "#7400B8"]
    }
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Accuracy bars
    fig.add_trace(go.Bar(
        x=data["Model"],
        y=data["Train Accuracy"],
        name='Train Accuracy',
        marker_color=data["Color"],
        text=data["Train Accuracy"],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>Train Accuracy: %{y}%<extra></extra>"
    ))
    
    fig.add_trace(go.Bar(
        x=data["Model"],
        y=data["Val Accuracy"],
        name='Val Accuracy',
        marker_color=data["Color"],
        opacity=0.7,
        text=data["Val Accuracy"],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>Val Accuracy: %{y}%<extra></extra>"
    ))
    
    # Add parameter count as line chart
    fig.add_trace(go.Scatter(
        x=data["Model"],
        y=data["Params (M)"],
        name='Parameters (M)',
        mode='lines+markers',
        yaxis='y2',
        line=dict(color='#FFFFFF', width=3),
        marker=dict(size=8, color='#FFFFFF'),
        hovertemplate="<b>%{x}</b><br>Parameters: %{y}M<extra></extra>"
    ))
    
    fig.update_layout(
        title={
            'text': "<b>Model Performance Deep Dive</b><br><span style='font-size:0.9em'>Accuracy vs Parameters (Size)</span>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Model Architecture",
        yaxis_title="Accuracy (%)",
        yaxis2=dict(
            title="Parameters (Millions)",
            overlaying='y',
            side='right',
            range=[0, max(data["Params (M)"])*1.1]
        ),
        barmode='group',
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#e0e0e0"),
        height=550,
        xaxis=dict(gridcolor="#3a3d48", tickangle=-45),
        yaxis=dict(gridcolor="#3a3d48", range=[40,105]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=100)
    )
    
    # Add custom annotations for speed
    for i, model in enumerate(data["Model"]):
        fig.add_annotation(
            x=model,
            y=105,
            text=data["Training Speed"][i],
            showarrow=False,
            font=dict(size=12, color="#e0e0e0")
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
# Enhanced Insights & Recommendations
# Enhanced Insights & Recommendations
def insights_section():
    st.markdown("## 🔍 Deep Analysis & Strategic Recommendations")
    
    # Three columns with interactive tabs
    tab1, tab2, tab3 = st.tabs(["📉 Performance Insights", "⚙️ Architecture Tradeoffs", "🚀 Deployment Guide"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚀 Optimization Impact")
            st.metric("FFNN Improvement", "+29.3%", "val accuracy")
            st.metric("CNN Improvement", "+19.84%", "with tuning")
            st.metric("LSTM Improvement", "+2.36%", "from augmentation")
            
        with col2:
            st.markdown("### 🏆 Efficiency Leaders")
            st.metric("MobileNetV2", "99.07%", "val accuracy")
            st.metric("Optimized CNN", "93.67%", "with 2.4M params")
            st.metric("CNN-RNN Hybrid", "93.48%", "best hybrid")
        
        st.markdown("---")
        st.markdown("#### 🧠 Unexpected Discovery")
        st.info("""
        LSTM base performance (84.7%) was better than expected for image classification, 
        but showed **diminishing returns** from optimization compared to other architectures.
        """)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⚡ Computational Efficiency")
            st.markdown("**Training Speed Ratings** (same hardware)")
            st.progress(80, text="FFNN: ⚡⚡⚡⚡")
            st.progress(60, text="CNN: ⚡⚡⚡")
            st.progress(20, text="LSTM: ⚡")
            st.progress(70, text="MobileNetV2: ⚡⚡⚡ (after init)")
            
        with col2:
            st.markdown("### ⚖️ Accuracy vs Size Tradeoff")
            st.markdown("**Top Performers**")
            
            df = pd.DataFrame({
                'Model': ['MobileNetV2', 'Optimized CNN'],
                'Accuracy': [99.07, 93.67],
                'Params (M)': [3.5, 2.4]
            })
            
            fig = px.scatter(df, x="Params (M)", y="Accuracy", text="Model",
                            size=[20, 15], color=["#7400B8", "#5390D9"],
                            labels={"Params (M)": "Parameters (Millions)"})
            
            fig.update_traces(textposition='top center')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("**Key Insight:** Optimized CNN provides 94% of MobileNet's accuracy with 31% fewer parameters")
    
    with tab3:
        st.markdown("### 🏗️ Implementation Guide")
        
        cols = st.columns(3)
        with cols[0]:
            with st.container(border=True):
                st.markdown("#### ☁️ Cloud Deployment")
                st.metric("First Choice", "MobileNetV2")
                st.markdown("**Reason:** Maximize accuracy")
                st.markdown("**Cost:** Higher compute")
                
        with cols[1]:
            with st.container(border=True):
                st.markdown("#### 📱 Edge Devices")
                st.metric("First Choice", "Optimized CNN")
                st.markdown("**Reason:** Best accuracy/size")
                st.markdown("**Savings:** 31% smaller")
                
        with cols[2]:
            with st.container(border=True):
                st.markdown("#### 🔋 Constrained IoT")
                st.metric("First Choice", "FFNN")
                st.markdown("**Reason:** Fastest inference")
                st.markdown("**Tradeoff:** ~85% accuracy")
        
        st.markdown("---")
        st.markdown("#### ❌ When to Avoid")
        st.warning("""
        **LSTM architectures** are not recommended for pure image classification - their sequential processing 
        provides no accuracy advantage over CNNs while being significantly slower (3-4× training time).
        """)

# Home page
def home_page():
    animated_header()
    
    # Introduction section with cards - Dark mode compatible
    st.markdown("## 📌 Project Overview")
    cols = st.columns(3)
    with cols[0]:
        with st.container(border=True):
            st.markdown("### 🎯 Objective")
            st.markdown("<div style='color: white;'>Compare deep learning architectures for binary image classification of cats and dogs.</div>", unsafe_allow_html=True)
    with cols[1]:
        with st.container(border=True):
            st.markdown("### 🛠️ Approach")
            st.markdown("<div style='color: white;'>Tested FFNN, CNN, RNN variants, hybrids, and transfer learning with MobileNetV2.</div>", unsafe_allow_html=True)
    with cols[2]:
        with st.container(border=True):
            st.markdown("### 🏆 Results")
            st.markdown("<div style='color: white;'>MobileNetV2 achieved <strong>99.15% validation accuracy</strong>, outperforming all other architectures.</div>", unsafe_allow_html=True)
    
    # Dataset section with tabs - Dark mode compatible
    st.markdown("## 📂 Dataset Information")
    tab1, tab2, tab3 = st.tabs(["Overview", "Augmentation", "Samples"])
    
    with tab1:
        st.markdown("""
        <div style="padding: 20px; border-radius: 10px;">
            <h4 style="color: white;">Dataset Composition</h4>
            <ul style="color: white;">
                <li><b>Original Dataset:</b> 1,500 cats and 1,500 dogs (3,000 total images)</li>
                <li><b>After Augmentation:</b> 13,500 cats and 13,500 dogs (27,000 total images)</li>
                <li><b>Train/Val/Test Split:</b> 70%/15%/15%</li>
                <li><b>Image Sizes:</b> Tested from 32×32 to 96x96 pixels</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Augmentation Examples")
        
        # Original image
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### Original Image")
            try:
                original_cat = Image.open("cats_and_dogs/cats/cat.0.jpg")
                st.image(original_cat, caption="Original Cat", use_container_width=True)
            except FileNotFoundError:
                st.warning("Original cat image not found at: cats_and_dogs/cats/cat.0.jpg")
            
            try:
                original_dog = Image.open("cats_and_dogs/dogs/dog.0.jpg")
                st.image(original_dog, caption="Original Dog", use_container_width=True)
            except FileNotFoundError:
                st.warning("Original dog image not found at: cats_and_dogs/dogs/dog.0.jpg")
        
        with col2:
            st.markdown("#### Augmentation Variations")
            
            # Define augmentation types and their display names
            augmentations = {
                "zoom": "Zoom",
                "brightness": "Brightness",
                "contrast": "Contrast",
                "gaussian_blur": "Gaussian Blur",
                "vertical_flip": "Vertical Flip"
            }
            
            # Cat augmentations
            st.markdown("**Cat Augmentations**")
            cols = st.columns(len(augmentations))
            for col, (aug_key, aug_name) in zip(cols, augmentations.items()):
                with col:
                    try:
                        img = Image.open(f"augmented_dataset/cats/cat.0_{aug_key}.jpg")
                        st.image(img, caption=aug_name, use_container_width=True)
                    except FileNotFoundError:
                        st.warning(f"Augmented image not found: augmented_dataset/cats/cat.0_{aug_key}.jpg")
            
            # Dog augmentations
            st.markdown("**Dog Augmentations**")
            cols = st.columns(len(augmentations))
            for col, (aug_key, aug_name) in zip(cols, augmentations.items()):
                with col:
                    try:
                        img = Image.open(f"augmented_dataset/dogs/dog.0_{aug_key}.jpg")
                        st.image(img, caption=aug_name, use_container_width=True)
                    except FileNotFoundError:
                        st.warning(f"Augmented image not found: augmented_dataset/dogs/dog.0_{aug_key}.jpg")
            
            st.markdown("""
            <div style="background-color: rgba(30, 136, 229, 0.1); padding: 15px; border-radius: 10px; margin-top: 20px;">
                <h4 style="color: #6b8cff; margin-top: 0;">Augmentation Techniques Applied</h4>
                <ul style="color: #e0e0e0;">
                    <li><b>Zoom:</b> ±15% random zoom</li>
                    <li><b>Brightness:</b> 0.8-1.2x adjustment</li>
                    <li><b>Contrast:</b> 0.8-1.2x adjustment</li>
                    <li><b>Gaussian Blur:</b> Mild blurring (σ=0.5-1.5)</li>
                    <li><b>Vertical Flip:</b> 50% chance of vertical flip</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### Dataset Samples")
        
        # Display original samples
        st.markdown("#### Original Samples")
        sample_cols = st.columns(4)
        sample_images = [
            "cats_and_dogs/cats/cat.0.jpg",
            "cats_and_dogs/dogs/dog.0.jpg",
            "cats_and_dogs/cats/cat.1.jpg",
            "cats_and_dogs/dogs/dog.1.jpg"
        ]
        for col, img_path in zip(sample_cols, sample_images):
            with col:
                try:
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                except FileNotFoundError:
                    st.warning(f"Sample image not found: {img_path}")
        
        # Display augmented samples
        st.markdown("#### Augmented Samples")
        sample_cols = st.columns(4)
        sample_images = [
            "augmented_dataset/cats/cat.0_zoom.jpg",
            "augmented_dataset/dogs/dog.0_brightness.jpg",
            "augmented_dataset/cats/cat.1_contrast.jpg",
            "augmented_dataset/dogs/dog.1_vertical_flip.jpg"
        ]
        for col, img_path in zip(sample_cols, sample_images):
            with col:
                try:
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                except FileNotFoundError:
                    st.warning(f"Augmented sample not found: {img_path}")

    # Key Findings section - Dark mode compatible
    st.markdown("## 🔍 Key Findings")
    
    with st.expander("📊 Input Size Analysis", expanded=True):
        size_data = {
            "Input Size": ["32×32", "32×32", "64×64", "64×64", "128×128"],
            "Channels": ["RGB (3)", "Grayscale (1)", "RGB (3)", "Grayscale (1)", "RGB (3)"],
            "Test Accuracy": [74.7, 69.8, 74.3, 71.3, 71.1]
        }
        
        fig = px.bar(size_data, 
                     x="Input Size", 
                     y="Test Accuracy", 
                     color="Channels",
                     barmode="group",
                     title="Accuracy by Input Size and Color Channels",
                     text="Test Accuracy",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_layout(
            yaxis_range=[65,80],
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Architecture comparison
    st.markdown("## 🏗️ Model Architectures")
    arch_tabs = st.tabs(["FFNN", "CNN", "RNN Variants", "MobileNetV2"])
    
    with arch_tabs[0]:
        st.markdown("""
        <div style="padding: 20px; border-radius: 10px;">
            <h4 style="color: white;">Feedforward Neural Network</h4>
            <div style="display: flex;">
                <div style="flex: 1;">
                    <img src="https://miro.medium.com/max/700/1*oB3S5yHHhvougJkPXuc8og.gif" 
                         style="width: 100%; border-radius: 8px;">
                </div>
                <div style="flex: 1; padding-left: 20px;">
                    <h5 style="color: white;">Architecture Details</h5>
                    <ul style="color: white;">
                        <li><b>Input:</b> Flattened 32×32×3 image (3,072 features)</li>
                        <li><b>Hidden Layers:</b> 3 dense layers (512, 256, 128 neurons)</li>
                        <li><b>Regularization:</b> 30% Dropout</li>
                        <li><b>Activation:</b> ReLU (hidden), Softmax (output)</li>
                    </ul>
                    <h5 style="color: white;">Performance</h5>
                    <ul style="color: white;">
                        <li>Best Accuracy: 70.22% (after augmentation)</li>
                        <li>Fastest training time</li>
                        <li>Prone to overfitting without augmentation</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with arch_tabs[1]:
        st.markdown("""
        <div style="padding: 20px; border-radius: 10px;">
            <h4 style="color: white;">Convolutional Neural Network</h4>
            <div style="display: flex;">
                <div style="flex: 1;">
                    <img src="https://i.postimg.cc/D0cCwVbT/cnn.png" 
                         style="width: 100%; border-radius: 8px;">
                </div>
                <div style="flex: 1; padding-left: 20px;">
                    <h5 style="color: white;">Architecture Details</h5>
                    <ul style="color: white;">
                        <li><b>Convolutional Blocks:</b> Two blocks (16 and 32 filters)</li>
                        <li><b>Pooling:</b> MaxPooling after each block</li>
                        <li><b>Regularization:</b> 20% Dropout</li>
                        <li><b>Classifier:</b> Dense layer (128 neurons)</li>
                    </ul>
                    <h5 style="color: white;">Performance</h5>
                    <ul style="color: white;">
                        <li>Best Accuracy: 92.57% (after augmentation)</li>
                        <li>Excellent balance of speed and accuracy</li>
                        <li>Significantly benefits from augmentation</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with arch_tabs[2]:  # RNN Variants tab
        rnn_tabs = st.tabs(["LSTM", "CNN-RNN Hybrid"])
    
    with rnn_tabs[0]:  # LSTM
        st.markdown("""
        <div style="padding: 20px; border-radius: 10px;">
            <h4 style="color: white;">LSTM Architecture</h4>
                <div style="flex: 1; padding-left: 20px;">
                    <h5 style="color: white;">Architecture Breakdown</h5>
                    <ul style="color: white;">
                        <li><b>Bidirectional LSTM:</b> Processes sequences in both directions</li>
                        <li><b>Memory Cells:</b> Better long-term dependency handling</li>
                        <li><b>Classifier:</b> Dense layer for final classification</li>
                    </ul>
                    <h5 style="color: white;">Performance</h5>
                    <ul style="color: white;">
                        <li>Initial Accuracy: 84.70%</li>
                        <li>After Augmentation & Tuning: 87.06%</li>
                        <li>Better at capturing long-range patterns</li>
                        <li>More computationally expensive than other architectures</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with rnn_tabs[1]:  # CNN-RNN Hybrid
        st.markdown("""
        <div style="padding: 20px; border-radius: 10px;">
            <h4 style="color: white;">CNN-RNN Hybrid Architecture</h4>
            <div style="display: flex;">
                <div style="flex: 1; padding-left: 20px;">
                    <h5 style="color: white;">Architecture Breakdown</h5>
                    <ul style="color: white;">
                        <li><b>CNN Frontend:</b> Spatial feature extraction</li>
                        <li><b>LSTM Backend:</b> Sequential processing of features</li>
                        <li><b>Combined:</b> Leverages strengths of both architectures</li>
                    </ul>
                    <h5 style="color: white;">Performance</h5>
                    <ul style="color: white;">
                        <li>Initial Accuracy: 90.02%</li>
                        <li>After Augmentation & Tuning: 93.48%</li>
                        <li>Excellent spatial and temporal modeling</li>
                        <li>More complex than standalone architectures</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with arch_tabs[3]:
        st.markdown("""
        <div style="padding: 20px; border-radius: 10px;">
            <h4 style="color: white;">MobileNetV2 Architecture</h4>
            <div style="display: flex;">
                <div style="flex: 1;">
                    <img src="https://production-media.paperswithcode.com/models/mobilenetv2.png" 
                         style="width: 100%; border-radius: 8px;">
                </div>
                <div style="flex: 1; padding-left: 20px;">
                    <h5 style="color: white;">Architecture Details</h5>
                    <ul style="color: white;">
                        <li><b>Base Model:</b> Pretrained MobileNetV2 (ImageNet)</li>
                        <li><b>Feature Extraction:</b> 1280-dimensional feature vector</li>
                        <li><b>Classifier:</b> Custom dense layers on top</li>
                        <li><b>Fine-tuning:</b> Last 20 layers unfrozen</li>
                    </ul>
                    <h5 style="color: white;">Performance</h5>
                    <ul style="color: white;">
                        <li>Best Accuracy: 99.15%</li>
                        <li>Fast convergence with transfer learning</li>
                        <li>Excellent generalization capability</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Performance comparison - Dark mode compatible
    st.markdown("## 📈 Performance Comparison")
    performance_chart()
    insights_section()
    
    # Conclusion
    st.markdown("## 🎯 Conclusion")
    st.markdown("""
    <div style="background-color: rgba(30, 136, 229, 0.2); padding: 20px; border-radius: 10px; border-left: 5px solid #1e88e5;">
        <p style="font-size: 16px; color: white;">
            Our comprehensive evaluation demonstrates that <b>transfer learning with MobileNetV2</b> provides 
            the best performance for cats vs dogs classification, achieving near-perfect accuracy while 
            maintaining reasonable computational requirements.
        </p>
        <p style="font-size: 16px; color: white;">
            For resource-constrained environments, a <b>custom CNN with augmentation</b> offers an excellent 
            balance between accuracy and efficiency.
        </p>
        <p style="font-size: 16px; font-weight: bold; color: #1e88e5;">
            Try out our best model in the Test section! 🚀
        </p>
    </div>
    """, unsafe_allow_html=True)

# Test page
# Test page
def test_page():
    animated_header()
    
    # Add some fun animation
    if st.session_state.get('first_visit', True):
        rain(emoji="🐾", font_size=20, falling_speed=5)
        st.session_state.first_visit = False
    
    st.markdown("""
    <div class="custom-card">
        <h3 style="color: #2c3e50;">🔍 How It Works</h3>
        <ol>
            <li>Upload one or more images of cats or dogs</li>
            <li>Our MobileNetV2 model will analyze the images</li>
            <li>View predictions with confidence levels</li>
            <li>Explore detailed probability distributions</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload section
    option = st.radio("Select upload option:", 
                     ("Single Image", "Multiple Images"),
                     horizontal=True)
    
    if option == "Single Image":
        uploaded_file = st.file_uploader("Choose an image...", 
                                       type=["jpg", "jpeg", "png", "webp"],
                                       label_visibility="collapsed")
        
        if uploaded_file is not None and model is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📤 Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Your Image", use_container_width=True)
            
            with col2:
                with st.spinner("🔮 Predicting..."):
                    class_name, confidence, probs = predict_image(image, model)
                
                st.markdown("### 📊 Prediction Results")
                
                # Confidence display
                confidence_class = "high-confidence" if confidence > 0.85 else "moderate-confidence" if confidence > 0.65 else "low-confidence"
                st.markdown(f"""
                <div class="prediction-result {confidence_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.1rem;">Prediction: 
                                <strong style="color: {'#6b8cff' if class_name=='dog' else '#ff6b6b'}">
                                    {class_name.capitalize()}
                                </strong>
                        </span>
                        <span style="font-family: monospace; font-size: 1rem;">
                            {confidence*100:.1f}% confidence
                        </span>
                    </div>
                    <progress value="{confidence}" max="1"></progress>
                </div>
                """, unsafe_allow_html=True)

                
                # Probability chart
                prob_data = {
                    "Class": ["Cat", "Dog"],
                    "Probability": [probs[0], probs[1]]
                }
                
                fig = px.bar(prob_data, 
                            x="Class", 
                            y="Probability",
                            color="Class",
                            color_discrete_map={"Cat": "#FF9AA2", "Dog": "#A2D2FF"},
                            text_auto='.1%',
                            range_y=[0, 1])
                
                fig.update_traces(textposition='outside')
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Fun feedback
                if class_name == "cat":
                    st.markdown("""
                    <div style="background-color: #fff0f0; padding: 15px; border-radius: 10px; border-left: 5px solid #ff6b6b;">
                        <p style="margin: 0;">🐱 <strong>Meow!</strong> Our model thinks this is a cat.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; border-left: 5px solid #6b8cff;">
                        <p style="margin: 0;">🐶 <strong>Woof!</strong> Our model thinks this is a dog.</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:  # Multiple Images
        uploaded_files = st.file_uploader("Choose multiple images...", 
                                        type=["jpg", "jpeg", "png", "webp"],
                                        accept_multiple_files=True)
        
        if uploaded_files and model is not None:
            st.toast(f"Processing {len(uploaded_files)} images...", icon="⏳")
            
            batch_size = 4
            for i in range(0, len(uploaded_files), batch_size):
                batch_files = uploaded_files[i:i+batch_size]
                cols = st.columns(len(batch_files))
                
                for idx, (col, uploaded_file) in enumerate(zip(cols, batch_files)):
                    with col:
                        image = Image.open(uploaded_file)
                        st.image(image, use_container_width=True)
                        
                        with st.spinner(""):
                            class_name, confidence, _ = predict_image(image, model)
                        
                        if confidence > 0.9:
                            badge_color = "#4CAF50"
                            emoji = "🐾"
                        elif confidence > 0.7:
                            badge_color = "#FFC107"
                            emoji = "🤔"
                        else:
                            badge_color = "#F44336"
                            emoji = "❓"
                        
                        st.markdown(f"""
                        <div style="background-color: {badge_color}20; 
                                    padding: 10px; 
                                    border-radius: 8px; 
                                    border-left: 4px solid {badge_color};
                                    margin-top: 5px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-weight: bold; color: {badge_color};">
                                    {class_name.capitalize()}
                                </span>
                                <span style="font-size: 0.8em; color: #666;">
                                    {confidence*100:.1f}% {emoji}
                                </span>
                            </div>
                            <progress value="{confidence}" max="1" 
                                      style="width: 100%; height: 6px;">
                            </progress>
                        </div>
                        """, unsafe_allow_html=True)
# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center;">
        <h1 style="font-size: 1.8rem; color: #2c3e50; margin-bottom: 0;">🐾 Cats & Dogs</h1>
        <p style="color: #7f8c8d; margin-top: 0;">Deep Learning Showcase</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio("Navigate", 
                   ["🏠 Home", "🔍 Test the Model"],
                   label_visibility="collapsed")
    
    st.markdown("---")
    
    st.markdown("### Model Information")
    st.markdown("""
    - **Architecture:** MobileNetV2
    - **Accuracy:** 99.15%
    - **Input Size:** 96x96px
    """)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.8em;">
        <p>---</p>
        <p>Computer and Control Department - Faculty of Engineering - Suez Canal University</p>
    </div>
    """, unsafe_allow_html=True)

# Page routing
if "Home" in page:
    home_page()
else:
    test_page()