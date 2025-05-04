import pickle

# Load the model and encoders
model = pickle.load(open('model/price_model.pkl', 'rb'))
brand_encoder = pickle.load(open('model/brand_encoder.pkl', 'rb'))
category_encoder = pickle.load(open('model/category_encoder.pkl', 'rb'))

# Define input
user_input = {
    'Category': 'Fitness & Health',
    'Brand': 'FitLife',
    'Average_Competitor_Price': 60,
    'Ratings': 4.5,
    'Reviews': 150,
    'Launch_Year': 2024
}

# Encode inputs
category = category_encoder.transform([user_input['Category']])[0]
brand = brand_encoder.transform([user_input['Brand']])[0]

X = [[category, brand, user_input['Average_Competitor_Price'],
      user_input['Ratings'], user_input['Reviews'], user_input['Launch_Year']]]

# Predict
predicted_price = model.predict(X)[0]
print(f"Predicted Price: ${predicted_price:.2f}")
