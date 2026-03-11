import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------

st.markdown("""
<style>

.main-title {
    font-size: 40px;
    font-weight: 700;
}

.result-box {
    padding:20px;
    border-radius:12px;
    background:#111;
    border:1px solid #333;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Models
# -----------------------------

category_model = pickle.load(open("category_model.pkl","rb"))
priority_model = pickle.load(open("priority_model.pkl","rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl","rb"))

# -----------------------------
# Load Dataset (for dashboard)
# -----------------------------

df = pd.read_csv("customer_support_tickets.csv")

# -----------------------------
# Sidebar Navigation
# -----------------------------

st.sidebar.title("🎫 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Ticket Classifier",
        "Example Tickets",
        "Dataset Dashboard",
        "About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Future Interns — Machine Learning Task 2")

# -----------------------------
# Example Tickets
# -----------------------------

example_tickets = [
    "I cannot log into my account after resetting my password.",
    "I was charged twice for my monthly subscription.",
    "The mobile app crashes whenever I open the dashboard.",
    "My order has not been delivered yet.",
    "The website shows an error when I try to checkout.",
    "Please help me update my billing information.",
    "My payment was successful but my order is not confirmed.",
    "The application is extremely slow and keeps freezing."
]

# -----------------------------
# PAGE 1 — CLASSIFIER
# -----------------------------

if page == "Ticket Classifier":

    st.title("🎫 Support Ticket Classification System")

    st.write("""
This application automatically **classifies support tickets and predicts their priority level**
using **Machine Learning and Natural Language Processing**.

Developed for **Future Interns – Machine Learning Task 2**.
""")

    st.divider()

    # Random ticket button
    if st.button("🎲 Try Random Ticket Example"):
        st.session_state["ticket_input"] = random.choice(example_tickets)

    ticket = st.text_area(
        "Enter Support Ticket Description",
        key="ticket_input",
        placeholder="Example: I cannot log into my account after resetting my password"
    )

    if st.button("🔍 Classify Ticket"):

        if ticket.strip() == "":
            st.warning("Please enter a ticket description.")
        else:

            vector = vectorizer.transform([ticket])

            category = category_model.predict(vector)[0]
            priority = priority_model.predict(vector)[0]

            st.divider()
            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"""
                    <div class="result-box">
                    <h3>📂 Ticket Category</h3>
                    <p style="font-size:22px">{category}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    f"""
                    <div class="result-box">
                    <h3>⚡ Ticket Priority</h3>
                    <p style="font-size:22px">{priority}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# -----------------------------
# PAGE 2 — EXAMPLE TICKETS
# -----------------------------

elif page == "Example Tickets":

    st.title("💡 Example Support Tickets")

    st.write("Try these example inputs in the classifier:")

    for e in example_tickets:
        st.write("•", e)

# -----------------------------
# PAGE 3 — DATASET DASHBOARD
# -----------------------------

elif page == "Dataset Dashboard":

    st.title("📊 Dataset Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Tickets", len(df))
    col2.metric("Ticket Categories", df["Ticket Type"].nunique())
    col3.metric("Priority Levels", df["Ticket Priority"].nunique())

    st.divider()

    st.subheader("Ticket Category Distribution")

    fig, ax = plt.subplots()
    sns.countplot(x="Ticket Type", data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.divider()

    st.subheader("Ticket Priority Distribution")

    fig2, ax2 = plt.subplots()
    sns.countplot(x="Ticket Priority", data=df, ax=ax2)
    st.pyplot(fig2)

# -----------------------------
# PAGE 4 — ABOUT PROJECT
# -----------------------------

elif page == "About Project":

    st.title("ℹ️ About This Project")

    st.markdown("""
### Support Ticket Classification System

This project builds a **machine learning system that automatically classifies customer support tickets and predicts their priority level**.

### Machine Learning Workflow

1. Data Cleaning  
2. NLP Text Preprocessing  
3. TF-IDF Feature Extraction  
4. Model Training  
5. Ticket Category Prediction  
6. Priority Prediction  

### Models Used

- Naive Bayes  
- Logistic Regression  
- Support Vector Machine (SVM)

### Goal

To help organizations automatically **route and prioritize customer support tickets efficiently**.

---

Developed for **Future Interns – Machine Learning Task 2**
""")
