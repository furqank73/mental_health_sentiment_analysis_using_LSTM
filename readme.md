# 🧠 Mental Health Sentiment Analysis using LSTM

This project applies **Deep Learning (LSTM - Long Short-Term Memory networks)** to analyze sentiments from mental health–related text data.  
It aims to classify text into categories like **Normal**, **Depression**, **Anxiety**, etc., based on the dataset.

---

## 📂 Dataset
- **File:** `Combined Data.csv`  
- Contains text samples related to mental health along with labeled sentiments.  
- Preprocessing is performed to clean the text (remove punctuation, stopwords, numbers, and convert to lowercase).

---

## 🛠️ Project Workflow
1. **Data Preprocessing**
   - Text cleaning (lowercasing, removing special characters, stopwords, and extra spaces).
   - Tokenization and padding sequences for LSTM input.
   - Label encoding.

2. **Model Building**
   - Uses an **LSTM network** built with TensorFlow/Keras.
   - Embedding layer for word representation.
   - LSTM layers with dropout to prevent overfitting.
   - Dense layer with softmax activation for multi-class sentiment classification.

3. **Model Training**
   - Optimizer: Adam  
   - Loss: Categorical Crossentropy  
   - Metrics: Accuracy  

4. **Evaluation**
   - Accuracy, Precision, Recall, F1 Score.
   - Confusion matrix visualization.

5. **Prediction**
   - Users can input text to predict mental health sentiment.

---

## 📊 Example Prediction
```python
text = "I feel well and looking for good things."
prediction = model.predict([text])
print("Predicted Sentiment:", prediction)
````

Output:

```
Predicted Sentiment: Anxiety
```

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/furqank73/mental_health_sentiment_analysis_using_LSTM.git
   cd mental_health_sentiment_analysis_using_LSTM
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run Jupyter Notebook:

   ```bash
   jupyter notebook sentiment_analysis.ipynb
   ```

---

## 📌 Future Improvements

* Add **Bidirectional LSTM** for better performance.
* Use **pre-trained embeddings** (GloVe, Word2Vec, BERT).
* Deploy as a **Streamlit Web App** for live predictions.

---

## 👨‍💻 Author

**M Furqan Khan**

* [GitHub](https://github.com/furqank73)
* [Kaggle](https://www.kaggle.com/fkgaming)
* [LinkedIn](https://www.linkedin.com/in/furqan-khan-256798268/)

