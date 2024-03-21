from flask import Flask, request, render_template, jsonify, session
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from flask_pymongo import PyMongo
from bson import ObjectId
import bcrypt

app = Flask(__name__)
app.secret_key = b'a1d5376930deb474939b8bcf'

app.config["MONGO_URI"] = "mongodb://localhost:27017/Login"
mongo = PyMongo(app)

# Load your dataset here
def load_data():
#    file_path = 'C:\Users\Angel\Desktop\EMPATHY\Test\empathy-food-waste\recipes_ingredients.csv'  # Update this path
    # df = pd.read_csv(r'D:\DLSU\PM\4th Year - Term 2 (23-24)\EMPATHY\Datasets\recipes_ingredients.csv')
#    df = pd.read_csv(file_path)
    
#    df = pd.read_csv(r'D:\DLSU\PM\4th Year - Term 2 (23-24)\EMPATHY\Datasets\recipes_ingredients.csv')
    df = pd.read_csv(r'C:\Users\Angel\Desktop\EMPATHY\Test\empathy-food-waste\recipes_ingredients.csv')
    # df = pd.read_csv(r'C:\Users\3515\Downloads\empathy\empathy-food-waste\recipes_ingredients.csv')
    
    # Correctly interpret 'ingredients', 'procedures', and 'tags' columns as lists
    df['ingredients'] = df['ingredients'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
    df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
#    df['steps'] = df['steps'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else []) 

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
    login_username = session.get("username", None)

    # Then, check if login_username is None (or your specified default) to determine if it exists.
    if login_username is None:
        return render_template('home.html')
    else:
        user = mongo.db.users.find_one({"name": login_username})

        food_preferences = user.get('food_preferences', [])
        dietary_preferences = user.get('dietary_preferences', [])
        allergens = user.get('allergens', [])
    
    #    food_preferences = user['food_preferences']
    #    dietary_preferences = user['dietary_preferences']
    #    allergens = user['allergens']

        return render_template('home.html', food_preferences=food_preferences, dietary_preferences=dietary_preferences, allergens=allergens)  # Create an index.html template for your form

# Edit User Page -- MICH
@app.route('/edituser')
def edituser():
    login_username = session.get("username", None)

    # Then, check if login_username is None (or your specified default) to determine if it exists.
    if login_username is None:
        return render_template('registerLogin.html')
    else:
       user = mongo.db.users.find_one({"name": login_username})

       food_preferences = user.get('food_preferences', [])
       dietary_preferences = user.get('dietary_preferences', [])
       allergens = user.get('allergens', [])
    
    #   food_preferences = user['food_preferences']
    #   dietary_preferences = user['dietary_preferences']
    #   allergens = user['allergens']
       
       return render_template('edituser.html', food_preferences=food_preferences, dietary_preferences=dietary_preferences, allergens=allergens)
    

# Update User Information
@app.route('/update_user', methods=['POST'])
def update_user():
    # Retrieve form data
    username = request.form['username']
    new_password = request.form['newpassword']
    confirm_password = request.form['confpassword']
    email = request.form['Email']

    # Retrieve user preferences from the form data
    food_preferences = request.form.getlist('foodPreference')
    dietary_preferences = request.form.getlist('dietaryPreference')
    allergens = request.form.getlist('allergen')

    food_preferences = [x.lower() for x in food_preferences]
    dietary_preferences = [x.lower() for x in dietary_preferences]
    allergens = [x.lower() for x in allergens]

    # Update user information in the database or perform other necessary actions
    try:
        if username == "" or new_password == "" or confirm_password == "" or email == "" :
            oldusername = session['username']
            # Update user information in the database
            mongo.db.users.update_one(
                {"name": oldusername},
                {"$set": {
                    "food_preferences": food_preferences,
                    "dietary_preferences": dietary_preferences,
                    "allergens": allergens
                }}
            )
        else:
            # Validate form data (for example, check if passwords match)
            if new_password != confirm_password:
                return "Passwords do not match. Please try again."
    
            # Hash the new password before updating
            salt_rounds = 15
            hashed_password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt(salt_rounds))
            
            # Update user information in the database
            mongo.db.users.update_one(
                {"name": username},
                {"$set": {
                    "password": hashed_password.decode(),
                    "email": email,
                    "food_preferences": food_preferences,
                    "dietary_preferences": dietary_preferences,
                    "allergens": allergens
                }}
            )

        print(food_preferences)
        print(dietary_preferences)
        print(allergens)

        # Redirect the user back to the edituser page or any other appropriate page
        return render_template('edituser.html', food_preferences=food_preferences, dietary_preferences=dietary_preferences, allergens=allergens)

    except Exception as e:
        print("Error updating user information:", e)
        return "Error updating user information.", 500

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

    # Remove commas after periods in steps
    if 'steps' in row_data:
        row_data['steps'] = row_data['steps'].replace('. ,', '. ')

    # Remove brackets from steps
    if 'steps' in row_data:
        row_data['steps'] = row_data['steps'].replace('[', '').replace(']', '')
    
    return render_template('recipepage.html', row=row_data)

@app.route('/registerLogin')
def registerLogin():
    
    return render_template('registerLogin.html')

@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.json
        
        username = data["username"]
        regpassword = data["regpassword"]
        regpassword2 = data["regpassword2"]
        email = data["Email"]

        existing_user = mongo.db.users.find_one({"name": username})
        existing_email = mongo.db.users.find_one({"email": email})

        response = {}

        if existing_user:
            response["usernameExists"] = True
        else: 
            response["usernameExists"] = False

        if existing_email:
            response["emailExists"] = True
        else:
            response["emailExists"] = False

        if regpassword != regpassword2:
            response["passwordsMatch"] = False
        else:
            response["passwordsMatch"] = True

        if len(regpassword) < 8:  
            response["passwordLength"] = False
        else:
            response["passwordLength"] = True

        if not(response.get("usernameExists")) and not(response.get("emailExists")) and response.get("passwordsMatch") and response.get("passwordLength"):
            response["success"] = True
            salt_rounds = 15
            hashed_password = bcrypt.hashpw(regpassword.encode(), bcrypt.gensalt(salt_rounds))
            mongo.db.users.insert_one({"name": username, "password": hashed_password.decode(), "email": email})
            session['username'] = username
            
        return jsonify(response), 200    

    except Exception as e:
        print("Error registering user:", e)
        return "Error registering user.", 500

@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.json
        
        login_username = data["loginusername"]
        login_password = data["loginpassword"]

        user = mongo.db.users.find_one({"name": login_username})
        response = {}

        if not user:
            response["usernameNotFound"] = True
        else:
            is_password_match = bcrypt.checkpw(login_password.encode(), user["password"].encode())
            if is_password_match:
                response["success"] = True
                session['username'] = login_username
            else:
                response["incorrectPassword"] = True

        return jsonify(response), 200
        
    except Exception as e:
        print("Error logging in:", e)
        return "Error logging in.", 500

@app.route('/logout')
def logout():
    # Clear the session data
    session.clear()
    # Redirect the user to the home page or any other appropriate page
    return render_template('Mainpage.html')


@app.route("/debug")
def debug():
    return jsonify(session)
    

if __name__ == '__main__':
    app.run(debug=True)

