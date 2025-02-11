import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Define the 101 class labels
labels = [
    'apple pie', 'baby back ribs', 'baklava', 'beef carpaccio', 'beef tartare',
    'beet salad', 'beignets', 'bibimbap', 'bread pudding', 'breakfast burrito',
    'bruschetta', 'caesar salad', 'cannoli', 'caprese salad', 'carrot cake',
    'ceviche', 'cheese plate', 'cheesecake', 'chicken curry', 'chicken quesadilla',
    'chicken wings', 'chocolate cake', 'chocolate mousse', 'churros', 'clam chowder',
    'club sandwich', 'crab cakes', 'creme brulee', 'croque madame', 'cup cakes',
    'deviled eggs', 'donuts', 'dumplings', 'edamame', 'eggs benedict', 'escargots',
    'falafel', 'filet mignon', 'fish and chips', 'foie gras', 'french fries',
    'french onion soup', 'french toast', 'fried calamari', 'fried rice',
    'frozen yogurt', 'garlic bread', 'gnocchi', 'greek salad', 'grilled cheese sandwich',
    'grilled salmon', 'guacamole', 'gyoza', 'hamburger', 'hot and sour soup',
    'hot dog', 'huevos rancheros', 'hummus', 'ice cream', 'lasagna', 'lobster bisque',
    'lobster roll sandwich', 'macaroni and cheese', 'macarons', 'miso soup', 'mussels',
    'nachos', 'omelette', 'onion rings', 'oysters', 'pad thai', 'paella', 'pancakes',
    'panna cotta', 'peking duck', 'pho', 'pizza', 'pork chop', 'poutine', 'prime rib',
    'pulled pork sandwich', 'ramen', 'ravioli', 'red velvet cake', 'risotto', 'samosa',
    'sashimi', 'scallops', 'seaweed salad', 'shrimp and grits', 'spaghetti bolognese',
    'spaghetti carbonara', 'spring rolls', 'steak', 'strawberry shortcake', 'sushi',
    'tacos', 'octopus balls', 'tiramisu', 'tuna tartare', 'waffles'
]

# Load the nutritional data from CSV
nutrition_file_path = "C:/Users/Harish/OneDrive/Desktop/ProjectFinal/Data/nutrition101.csv"
nutrition_data = pd.read_csv(nutrition_file_path)

# Load the custom-trained model
model_path = "C:/Users/Harish/OneDrive/Desktop/ProjectFinal/FastAPI_Backend/model_trained_101class.hdf5"
model = tf.keras.models.load_model(model_path)
st.success("Custom model successfully loaded!")

# Streamlit App Title
st.title("Food Image Recognition")

# File Uploader
uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

# Predict Function
def predict_food(image_file):
    # Preprocess the image
    img = Image.open(image_file).convert('RGB')  # Convert to RGB
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_array)
    top_indices = np.argsort(predictions[0])[-3:][::-1]  # Get top 3 predictions

    # Collect top 3 results
    top_results = []
    for i in top_indices:
        food_label = labels[i]
        confidence = f"{predictions[0][i] * 100:.2f}%"
        top_results.append((food_label, confidence))

    # Fetch nutrition data for the top prediction
    top_food = top_results[0][0]
    nutrition_info = nutrition_data[nutrition_data['name'] == top_food]
    if not nutrition_info.empty:
        protein = nutrition_info['protein'].values[0]
        calcium = nutrition_info['calcium'].values[0]
        fat = nutrition_info['fat'].values[0]
        carbohydrates = nutrition_info['carbohydrates'].values[0]
        vitamins = nutrition_info['vitamins'].values[0]
    else:
        protein, calcium, fat, carbohydrates, vitamins = "N/A", "N/A", "N/A", "N/A", "N/A"

    return top_results, {
        "label": top_food,
        "protein": protein,
        "calcium": calcium,
        "fat": fat,
        "carbohydrates": carbohydrates,
        "vitamins": vitamins
    }

# Display Prediction Results
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Predict the image
    top_results, nutrition_data = predict_food(uploaded_file)
    
    # Display top 3 predictions
    st.write("### Top 3 Predictions:")
    for food, confidence in top_results:
        st.write(f"- **{food}**: {confidence}")

    # Display nutrition data for the top prediction
    st.write(f"### Nutrition Data for **{nutrition_data['label']}**")
    st.write(f"- **Protein**: {nutrition_data['protein']} g")
    st.write(f"- **Calcium**: {nutrition_data['calcium']} g")
    st.write(f"- **Fat**: {nutrition_data['fat']} g")
    st.write(f"- **Carbohydrates**: {nutrition_data['carbohydrates']} g")
    st.write(f"- **Vitamins**: {nutrition_data['vitamins']} g")
