# 🎫 Support Ticket Classification & Prioritization System

### 🚀 Future Interns – Machine Learning Task 2

This project implements a **Machine Learning system that automatically classifies customer support tickets and predicts their priority level** using **Natural Language Processing (NLP)** techniques.

The goal of this project is to help organizations **automatically categorize support tickets and identify urgent issues**, improving response time and customer support efficiency.

---

# 📌 Project Overview

Customer support teams receive a large number of tickets every day.  
Manually reviewing and prioritizing them can be time-consuming and inefficient.

This project uses **Machine Learning and NLP techniques** to automatically:

• Classify support tickets into categories  
• Predict ticket priority levels  
• Provide an interactive interface using Streamlit  

---

# 🧠 Technologies Used

| Technology | Purpose |
|------------|--------|
| Python | Programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| NLTK | Natural Language Processing |
| Scikit-learn | Machine learning models |
| TF-IDF | Text feature extraction |
| Matplotlib & Seaborn | Data visualization |
| Streamlit | Interactive web application |

---

# ⚙️ Machine Learning Workflow

The project follows a complete Machine Learning pipeline:

1. Data Loading  
2. Data Cleaning  
3. Text Preprocessing (NLP)  
4. Feature Extraction using TF-IDF  
5. Train-Test Split  
6. Model Training  
7. Model Evaluation  
8. Ticket Category Prediction  
9. Ticket Priority Prediction  

---

# 🤖 Machine Learning Models Used

The following models were trained and evaluated:

• Naive Bayes  
• Logistic Regression  
• Support Vector Machine (SVM)

These models learn patterns from support ticket descriptions to predict:

📂 Ticket Category  
⚡ Ticket Priority

---

# 🌐 Streamlit Web Application

A **Streamlit web application** was developed to provide an interactive interface for the machine learning model.

### Features of the App

• Support ticket classification  
• Priority prediction  
• Example ticket suggestions  
• Random ticket generator  
• Dataset visualization dashboard  
• Clean and user-friendly interface  

Users can enter a support ticket description and instantly receive predictions for:

```
Ticket Category  
Ticket Priority
```

---

## 💻 Running the Streamlit App Locally

This project includes a **Streamlit web application** that allows users to interact with the trained machine learning models.

The application is **not deployed online**, but it can easily be run locally on your machine.

### 🚀 Steps to Run the Application

1️⃣ Install the required dependencies

```
pip install -r requirements.txt
```

2️⃣ Start the Streamlit application

```
streamlit run app.py
```

3️⃣ Open the app in your browser

```
http://localhost:8503
```

### ✨ Features of the Application

🔹 Classifies support tickets into categories  
🔹 Predicts ticket priority levels  
🔹 Provides example ticket inputs  
🔹 Displays dataset insights and visualizations  

This interactive interface allows users to test the **Support Ticket Classification & Prioritization System** easily.

---

⚠️ **Note:**  
The application currently runs **locally** and is **not deployed online**.

# 📊 Dataset

The dataset contains customer support ticket information including:

| Column | Description |
|------|-------------|
| Ticket Subject | Short summary of the issue |
| Ticket Description | Detailed explanation |
| Ticket Type | Ticket category |
| Ticket Priority | Priority level |

This dataset is used to train machine learning models for **ticket classification and prioritization**.

---

# 📁 Project Structure

```
FUTURE_ML_02
│
├── app.py
├── support_ticket_classification.ipynb
├── customer_support_tickets.csv
├── category_model.pkl
├── priority_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md
```

---

# 📈 Future Improvements

Possible improvements for the system include:

• Using deep learning models such as **BERT**  
• Deploying the application on cloud platforms  
• Adding advanced analytics dashboards  
• Integrating with real customer support ticket systems  

---

# 👨‍💻 **SIRIKI TAMAN**

Developed as part of:

🎓 **Future Interns – Machine Learning Task 2**

Project Title:  
🎫 **Support Ticket Classification & Prioritization System**
