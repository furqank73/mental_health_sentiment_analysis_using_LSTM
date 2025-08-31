import streamlit as st
import numpy as np
import joblib
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="🧠 Sentiment Analysis Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# Load Model, Tokenizer, and Label Encoder
# ===============================
@st.cache_resource(show_spinner=False)
def load_ml_components():
    """Load ML components with caching and error handling"""
    try:
        model = load_model("sentiment_model.keras")
        tokenizer = joblib.load("tokenizer.pickle")
        le = joblib.load("labelencoder.pickle")
        return model, tokenizer, le
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, tokenizer, le = load_ml_components()

# Set maxlen dynamically from model
try:
    MAX_LEN = model.input_shape[1]
except Exception:
    MAX_LEN = 150  # fallback if unavailable

# ===============================
# Preprocessing Function
# ===============================
def preprocess(text: str) -> str:
    """Match training-time preprocessing (simplest: lowercase + strip).
       Add more cleaning if you used it during training."""
    return text.lower().strip()

# ===============================
# Prediction Function
# ===============================
def predict_sentiment(text):
    """Predict sentiment with confidence scores for all classes"""
    cleaned = preprocess(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    pred = model.predict(pad, verbose=0)[0]

    label_idx = int(np.argmax(pred))
    label = le.inverse_transform([label_idx])[0]
    return label, pred, float(pred[label_idx])

# ===============================
# Visualization Functions
# ===============================
def plot_confidence_scores(confidence_scores, labels):
    """Create a bar chart of confidence scores"""
    fig = px.bar(
        x=labels, 
        y=confidence_scores,
        labels={'x': 'Sentiment', 'y': 'Confidence'},
        title="Confidence Scores by Sentiment",
        color=labels,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(showlegend=False)
    return fig

def generate_wordcloud(text, sentiment):
    """Generate a word cloud from text"""
    if not text.strip():
        return None
        
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis' if sentiment.lower() == 'positive' else 
                 'Reds' if sentiment.lower() == 'negative' else 'Blues'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud for {sentiment} Sentiment')
    return fig

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.title("⚙️ Settings")
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.7,
        step=0.05,
        help="Set the minimum confidence level for predictions"
    )
    
    show_wordcloud = st.checkbox("Show Word Cloud", value=True)
    show_details = st.checkbox("Show Detailed Analysis", value=True)
    
    st.divider()
    st.markdown("### 📊 App Info")
    st.markdown("""
    This sentiment analysis app uses a deep learning model to classify text into:
    - 😊 Positive
    - 😠 Negative  
    - 😐 Neutral
    """)
    
    try:
        st.markdown(f"**Model:** {model.name.title()}")
        st.markdown(f"**Classes:** {', '.join(le.classes_)}")
    except:
        pass

# ===============================
# Main Interface
# ===============================
st.title("🧠 Sentiment Analysis Pro")
st.markdown("Analyze the sentiment of your text with advanced visualization and insights.")

tab1, tab2, tab3 = st.tabs(["📝 Single Analysis", "📊 Batch Analysis", "ℹ️ About"])

# -------------------------------
# Tab 1: Single Analysis
# -------------------------------
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area(
            "✍️ Enter your text:",
            height=150,
            placeholder="Type or paste your text here to analyze its sentiment..."
        )
    
    with col2:
        st.markdown("### 💡 Examples")
        example_texts = [
            "I absolutely love this product! It's amazing.",
            "This is the worst experience I've ever had.",
            "The meeting was scheduled for 3 PM tomorrow."
        ]
        
        for example in example_texts:
            if st.button(example, key=example, use_container_width=True):
                user_input = example
                st.rerun()
    
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing sentiment..."):
                time.sleep(0.5)
                
                label, all_confidences, top_confidence = predict_sentiment(user_input)
                
                st.subheader("📊 Results")
                
                if top_confidence < confidence_threshold:
                    st.warning(f"⚠️ Low confidence prediction ({top_confidence:.2%})")
                
                sentiment_emojis = {
                    "Positive": "😊", 
                    "Negative": "😠", 
                    "Neutral": "😐"
                }
                emoji = sentiment_emojis.get(label, "🤔")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"<h2 style='text-align: center;'>{emoji} {label}</h2>", 
                               unsafe_allow_html=True)
                    st.markdown(f"<h4 style='text-align: center; color: gray;'>Confidence: {top_confidence:.2%}</h4>", 
                               unsafe_allow_html=True)
                
                if show_details:
                    st.divider()
                    st.subheader("📈 Detailed Analysis")
                    
                    confidences_fig = plot_confidence_scores(all_confidences, le.classes_)
                    st.plotly_chart(confidences_fig, use_container_width=True)
                    
                    if show_wordcloud:
                        wordcloud_fig = generate_wordcloud(user_input, label)
                        if wordcloud_fig:
                            st.pyplot(wordcloud_fig)
        else:
            st.warning("⚠️ Please enter some text before predicting.")

# -------------------------------
# Tab 2: Batch Analysis
# -------------------------------
with tab2:
    st.subheader("Batch Sentiment Analysis")
    st.markdown("Upload a CSV file with a 'text' column to analyze multiple entries at once.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'text' not in df.columns:
                st.error("The CSV file must contain a 'text' column.")
            else:
                if st.button("Analyze Batch", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results, confidence_scores = [], []
                    
                    for i, text in enumerate(df['text']):
                        if pd.notna(text) and str(text).strip():
                            label, _, confidence = predict_sentiment(str(text))
                            results.append(label)
                            confidence_scores.append(confidence)
                        else:
                            results.append("Invalid")
                            confidence_scores.append(0)
                        
                        progress = (i + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {i+1} of {len(df)} entries...")
                    
                    df['sentiment'] = results
                    df['confidence'] = confidence_scores
                    
                    st.success("Analysis complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    sentiment_counts = df['sentiment'].value_counts()
                    
                    with col1:
                        st.metric("Positive", sentiment_counts.get("Positive", 0))
                    with col2:
                        st.metric("Negative", sentiment_counts.get("Negative", 0))
                    with col3:
                        st.metric("Neutral", sentiment_counts.get("Neutral", 0))
                    
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Sentiment Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(df)
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")

# -------------------------------
# Tab 3: About
# -------------------------------
with tab3:
    st.subheader("About This App")
    st.markdown("""
    This Sentiment Analysis App uses a deep learning model to classify text into positive, negative, or neutral sentiments.
    
    **Features:**
    - 📝 Single text analysis with detailed confidence scores
    - 📊 Batch processing for CSV files
    - 📈 Visualizations including confidence charts and word clouds
    - ⚙️ Customizable confidence threshold
    
    **Technical details:**
    - TensorFlow/Keras model (LSTM)
    - Streamlit for the UI
    - Plotly + WordCloud for visualizations
    """)
    
    st.divider()
    st.markdown("### 🛠️ Model Information")
    
    try:
        model_config = model.get_config()
        st.text(f"Model: {model.name}")
        st.text(f"Input shape: {model.input_shape}")
        st.text(f"Output shape: {model.output_shape}")
        st.text(f"Number of layers: {len(model.layers)}")
    except:
        st.text("Model details not available")

# ===============================
# Footer
# ===============================
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Sentiment Analysis Pro • Built with Streamlit • "
    "<a href='https://github.com/furqank73/mental_health_sentiment_analysis_using_LSTM' target='_blank'>GitHub</a>"
    "</div>", 
    unsafe_allow_html=True
)
