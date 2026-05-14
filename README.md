# 🛡️ YouTube Spam Comment Detector

> A machine learning application that leverages the Naive Bayes classification algorithm to automatically detect and filter spam comments on YouTube with high accuracy.

---

## 📖 Project Overview

With the rise of automated bots and spam on social platforms, comment sections can quickly become cluttered with malicious links and repetitive promotional content. This project tackles that problem using Natural Language Processing (NLP) and Machine Learning to classify YouTube comments as either "Spam" or "Genuine".

### 🚀 Performance
* **Model Accuracy:** **92.78%**
* **Algorithm:** Naive Bayes Classifier

---

## 🧰 Technology Stack

<p align="left">
  <a href="https://skillicons.dev">
    <img src="https://skillicons.dev/icons?i=python,pandas,scikit,numpy,jupyter" />
  </a>
</p>

* **Language:** Python
* **Libraries:** Scikit-Learn (ML Models), Pandas (Data Manipulation), NumPy (Numerical Operations)
* **Environment:** Jupyter Notebook / Python Scripts

---

## ✨ Key Features & Methodology

### 1. Dataset Handling
* Utilizes a labeled dataset of real YouTube comments, pre-categorized into spam and non-spam classes.

### 2. Data Preprocessing
Before feeding text to the machine learning model, the raw data undergoes strict cleaning:
* Lowercasing text for uniformity.
* Removal of special characters, URLs, and punctuation.
* Removal of stop words (common words that don't add meaning, like "the", "is", "at").

### 3. Feature Extraction
* Converts the cleaned text data into a numerical format that the machine learning algorithm can understand. 
* Uses techniques like **Count Vectorization** or **TF-IDF** (Term Frequency-Inverse Document Frequency) to evaluate the importance of words within the comments.

### 4. Classification
* Trains a **Naive Bayes** model on the extracted features. Naive Bayes is highly effective for text classification tasks due to its foundation in probability and assumption of feature independence.

---

## 💻 Getting Started (Local Setup)

### Prerequisites
* Python 3.8+
* pip (Python package manager)

### Installation & Execution

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/youtube-spam-detector.git](https://github.com/yourusername/youtube-spam-detector.git)
   cd youtube-spam-detector
