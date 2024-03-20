from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

app = Flask(__name__)

# Load your dataset here
def load_data():
    file_path = 'C:\\Users\\Solid State Drive\\Documents\\GitHub\\empathy-food-waste\\recipes_ingredients.csv'  # Update this path
    df = pd.read_csv(file_path)
    
    # Correctly interpret 'ingredients', 'procedures', and 'tags' columns as lists
    df['ingredients'] = df['ingredients'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
    df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
    df['procedures'] = df['procedures'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])


    # Combine ingredients and tags into a single string per recipe
    df['combined_features'] = df.apply(lambda row: ' '.join(row['ingredients']) + ' ' + ' '.join(row['tags']), axis=1)

    return df

df = load_data()

# Print first few rows of 'combined_features' to inspect the content
print("hereeeeeeeeee!")
print(df['combined_features'].head())

# Function to preprocess and vectorize dataset for recommendation
def preprocess_and_vectorize(df):
    # Vectorize combined features
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    return tfidf_matrix, vectorizer

# Check for empty or NaN entries
# print(df['combined_features'].isnull().sum())
# print((df['combined_features'] == '').sum())

tfidf_matrix, vectorizer = preprocess_and_vectorize(df)

# ----- ROUTES -----

# Home/Model Page
@app.route('/main')
def mainpage(): 
    return render_template('Mainpage.html')  # Create an index.html template for your form

# Home/Model Page
@app.route('/')
def home():
    return render_template('home.html')  # Create an index.html template for your form

# Edit User Page -- MICH
@app.route('/edituser')
def edituser():
    return render_template('edituser.html')  # Create an index.html template for your form

# Update User Information
@app.route('/update_user', methods=['POST'])
def update_user():
    # Retrieve form data
    username = request.form['username']
    new_password = request.form['newpassword']
    confirm_password = request.form['confpassword']
    email = request.form['Email']
    
    # Validate form data (for example, check if passwords match)
    if new_password != confirm_password:
        return "Passwords do not match. Please try again."
    
    # Update user information in the database or perform other necessary actions
    # will add here ! when nahanap ko na db AHSAHA

    # Redirect the user back to the edituser page or any other appropriate page
    return render_template('edituser.html')

# Recipe Recommendations Page
@app.route('/recommend', methods=['POST'])
def recommend():
    # Retrieve ingredients and preferences submitted by the user
    ingredients = request.form['ingredients']  # Text input for ingredients
    food_preferences = request.form.getlist('foodPreference')  # Checkbox selections
    dietary_preferences = request.form.getlist('dietaryPreference')
    allergens = request.form.getlist('allergen')

    # Process the inputs (for demonstration, simply printing them)
    print("Ingredients:", ingredients)
    print("Food Preferences:", food_preferences)
    print("Dietary Preferences:", dietary_preferences)
    print("Allergens:", allergens)

    # Split 'ingredients' string into a list where each item is trimmed of leading/trailing spaces
    ingredients_list = [ingredient.strip() for ingredient in ingredients.split(',')]

    # Combine all lists into one 
    all_preferences = ingredients_list + food_preferences + dietary_preferences + allergens

    # Concatenate all items into a single string, separated by spaces
    user_input = ' '.join(all_preferences)

    # user_input = request.form['user_input']  # Assuming you have an input field for user input in your form
    user_input_tfidf = vectorizer.transform([user_input])
    cosine_sim = cosine_similarity(user_input_tfidf, tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar recipes
    recipe_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[recipe_indices]

    # Convert the DataFrame to a list of dictionaries for JSON serialization
    recommendations_list = recommendations.to_dict(orient='records')
    
    # Use jsonify to return the list of dictionaries as JSON
    # return jsonify(recommendations=recommendations_list)

    return render_template('reciperecommend.html', data=recommendations_list)

# Individual Recipe Page -- MICH
@app.route('/recipe/<int:recipe_id>')
def show_recipe(recipe_id):
    row = df[df['id'] == recipe_id]
    # Convert the row to a dictionary
    row_data = row.iloc[0].to_dict()
    
    return render_template('recipepage.html', row=row_data)


if __name__ == '__main__':
    app.run(debug=True)

