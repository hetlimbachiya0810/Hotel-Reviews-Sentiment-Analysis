# Hotel Review Sentiment Analysis (Deep Learning Project)

## 🏨 Project Overview
This project builds a Deep Learning–based sentiment classifier that predicts whether a hotel review is Positive or Negative. A custom LSTM model is trained using Keras/TensorFlow, and a Streamlit GUI allows real-time predictions.

## 🧠 Model Architecture
- Embedding Layer  
- LSTM (128 units)
- Dropout
- LSTM (256 units)
- Dropout
- Dense Output Layer (2 logits, sigmoid activation)

## 🔡 Text Preprocessing
- Lowercasing  
- Cleaning repeated quotes  
- Building a vocabulary of words appearing more than 10 times  
- Converting words to integers using word_to_int  
- Padding all sequences to maxlen = 120  

## 📁 Files Included
- app.py  
- hotel-sentiments-model.pkl  
- word_to_int.pkl  
- requirements.txt  
- hotel-reviews.csv  
- README.md  

## 🌐 Streamlit App
Run:
```
streamlit run app.py
```

## 📦 Requirements
```
streamlit
tensorflow==2.19.0
keras==3.10.0
numpy
pandas
```

## 🙌 Contributors
Het Limbachiya
