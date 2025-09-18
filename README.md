# 🍽️Food Recommendation System
This is a smart food recommendation app built using Python. It uses Streamlit for the frontend and FastAPI for the backend.


### ✨What It Does
Personalized Diet Plans 🧑‍🍳: Get daily meal recommendations based on your age, weight, height, and fitness goals.

Custom Recipe Search 🔍: Find recipes by choosing the exact nutrition values (like calories and protein) and ingredients you want.

Identify Food by Photo 📸: Upload a picture of a meal, and the app tells you what dish it is and its nutritional info.

---

### 📂 Project Structure
```
Directory structure:
└── harish-v07-foodrecommendationsystem/
    ├── README.md
    ├── requirements.txt
    ├── Data/
    |   ├── dataset.csv
    │   └── nutrition101.csv
    ├── FastAPI_Backend/
    │   ├── best_model_101class.hdf5
    │   ├── main.py
    │   ├── model.py
    │   └── model_trained_101class.hdf5
    └── Streamlit_Frontend/
        ├── Generate_Recommendations.py
        ├── MainPage.py
        ├── ImageFinder/
        │   ├── __init__.py
        │   └── ImageFinder.py
        └── pages/
            ├── 1_Food_Recommendation.py
            ├── Calorie Identification.py
            └── Custom_Food_Recommendation.py
```
---

### 🚀 Getting Started
#### Clone the repo
```
git clone https://github.com/harish-v07/FoodRecommendationSystem.git
```
In the project root run:

### Requirements
```
pip install -r requirements.txt
```
#### Backend:
```
uvicorn main:app --reload --port 8080                 
```
#### Frontend:
```
streamlit run MainPage.py
```
