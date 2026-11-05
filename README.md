💳 CreditLens

✨ Explainable Credit Default Prediction using SHAP, LIME & Tree-Based Models

<div align="center">"Python" (https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
"Streamlit" (https://img.shields.io/badge/Streamlit-Application-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
"Machine Learning" (https://img.shields.io/badge/Machine-Learning-8A2BE2?style=for-the-badge)
"Explainable AI" (https://img.shields.io/badge/Explainable-AI-00A896?style=for-the-badge)
"Status" (https://img.shields.io/badge/Status-Academic%20Project-success?style=for-the-badge)

🔍 Predict. Explain. Understand.

Making Machine Learning credit-risk predictions more transparent, interpretable and understandable.

</div>---

🌟 About CreditLens

CreditLens is an Explainable Artificial Intelligence (XAI) powered credit default prediction application designed to estimate customer credit risk and explain the reasoning behind its predictions.

Traditional Machine Learning models can produce predictions without clearly explaining why a particular result was generated.

CreditLens addresses this challenge by combining Tree-Based Machine Learning algorithms with powerful Explainable AI techniques such as SHAP and LIME.

The application analyses customer profiles and repayment history, generates a model risk score, and provides meaningful explanations of the factors influencing the prediction.

«💡 Our goal is not just to predict credit default risk, but to make the prediction understandable.»

---

🎯 The Problem We Solve

Traditional credit-risk prediction systems often behave like black boxes.

They may provide a prediction without clearly explaining:

- Why a customer is classified as high or low risk.
- Which repayment behaviours influence the prediction.
- Which customer features contribute most to the result.
- How different factors affect the model's estimated risk.

Our solution: An explainable credit-risk prediction system that combines Machine Learning predictions with transparent feature-level explanations.

---

🚀 Key Features

Feature| Description
💳 Credit Risk Prediction| Predicts customer credit default risk using Machine Learning
🌳 Random Forest| Uses a tree-based model to generate predictions
📊 Model Performance| Displays model evaluation metrics and visualizations
🧠 SHAP Explainability| Explains individual feature contributions
🔍 LIME Explanations| Provides local explanations for individual predictions
📈 Feature Importance| Identifies influential features across evaluated customers
👥 Customer Analytics| Provides customer-related analytical insights
📄 Document Check| Supports document-related verification
💰 Financial Health Calculator| Provides financial health calculations
📚 Financial Awareness| Provides financial awareness information

---

🧠 Explainable AI – The Heart of CreditLens

1. SHAP – SHapley Additive exPlanations

SHAP explains how individual features contribute to a Machine Learning prediction.

What it provides:

- Individual customer prediction explanations.
- Positive and negative feature contributions.
- Overall feature importance.
- Visual representation of feature influence.

2. LIME – Local Interpretable Model-Agnostic Explanations

LIME explains individual predictions using a simpler local approximation of the original Machine Learning model.

What it provides:

- Local prediction explanations.
- Identification of influential input features.
- Positive and negative feature weights.
- Easier interpretation of complex model behaviour.

3. Tree-Based Machine Learning

Random Forest is used to analyse customer information and repayment patterns to estimate credit default risk.

---

⚙️ How CreditLens Works

flowchart TD
    A["👤 Customer Information"] --> B["📋 Credit History"]
    B --> C["✅ Input Validation"]
    C --> D["🌳 Machine Learning Model"]
    D --> E["📊 Credit Risk Score"]
    E --> F["🎯 Risk Classification"]
    F --> G["🧠 SHAP Explanation"]
    F --> H["🔍 LIME Explanation"]
    G --> I["📈 Feature Importance"]
    H --> J["💡 Local Explanation"]
    I --> K["Understand the Prediction"]
    J --> K

---

🛠️ Technology Stack

Category| Technologies
Programming Language| Python
Frontend / Interface| Streamlit
Data Processing| Pandas, NumPy
Machine Learning| Scikit-learn
Prediction Algorithm| Random Forest
Explainable AI| SHAP, LIME
Visualization| Plotly, Matplotlib

---

📊 Why Explainable AI?

Traditional ML| CreditLens
Gives predictions| Gives predictions with explanations
Limited transparency| Feature-level transparency
Difficult to interpret| Easier to understand
Focuses on prediction| Focuses on prediction and interpretation

---

🔄 Project Workflow

1. Collect customer profile information.
2. Enter the customer's recent repayment history.
3. Validate the entered information.
4. Process the data using the trained Machine Learning model.
5. Generate a credit risk score.
6. Classify the estimated risk.
7. Use SHAP to explain feature contributions.
8. Use LIME to generate local explanations.
9. Display visual insights to help understand the prediction.

---

💻 Installation & Setup

Prerequisites

- Python 3.10+
- Visual Studio Code
- Git

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL

2. Navigate to the project folder

cd YOUR_REPOSITORY_FOLDER

3. Create a virtual environment

python -m venv venv

4. Activate the environment

Windows:

venv\Scripts\activate

5. Install the required packages

pip install -r requirements.txt

6. Run CreditLens

streamlit run app.py

🎉 Your application will open in your browser!

---

🔮 Future Enhancements

- 🔐 Improved customer information validation.
- 📄 Enhanced document-based verification.
- 🤖 Evaluation of additional Machine Learning algorithms.
- 📊 Improved prediction and explanation visualizations.
- ☁️ Secure cloud deployment.
- 🔗 Integration with verified financial data sources.

---



<div align="center">💳 CREDITLENS

See the Prediction. Understand the Reason.

Bringing Transparency to Credit Risk Prediction through Explainable AI.

⭐ If you find this project interesting, consider giving the repository a star!

</div>
