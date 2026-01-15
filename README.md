🍴 Restaurant Recommender System
📌 Overview
This project is a content-based restaurant recommender system built with Python. It uses restaurant attributes such as cuisine type, average cost, and ratings to suggest similar restaurants that match user preferences. The system leverages machine learning techniques like one-hot encoding, feature scaling, and cosine similarity to generate personalized recommendations.

🚀 Features
- Data Preprocessing
- Cleans and standardizes dataset columns
- Handles missing values for cuisine, price, and rating
- Converts categorical cuisines into one-hot encoded vectors
- Scales numerical features (price, rating) for fair comparison
- Recommendation Engine
- Computes similarity between restaurants using cosine similarity
- Filters restaurants based on user preferences (cuisine, price range, minimum rating)
- Provides top-N recommendations ranked by similarity
- Flexible Functions
- recommend_for_user() → personalized recommendations based on filters
- recommend_by_reference() → find restaurants similar to a given one
- get_candidate_indices() → filter dataset by constraints
- Output Options
- Displays recommendations in the console
- Saves results to a CSV file for further use

🛠️ Tech Stack
- Python 3.9+
- Libraries:
- pandas – data manipulation
- numpy – numerical operations
- scikit-learn – preprocessing & similarity computation

📂 Dataset
The recommender expects a dataset with columns such as:
- Restaurant Name → renamed to name
- Cuisines → renamed to cuisine
- Average Cost for two → renamed to price
- Aggregate Rating → renamed to rating
⚠️ If your dataset uses different column names, update the renaming dictionary in the script accordingly.

▶️ Usage
1. Clone the repository:
git clone https://github.com/your-username/restaurant-recommender.git
cd restaurant-recommender
2.Install dependencies:
pip install -r requirements.txt
3. Run the recommender:
python restaurant_recommender.py
4.Example test case (inside the script):
recommendations = recommend_for_user(
    cuisine_pref="Italian",
    price_min=250,
    price_max=500,
    min_rating=4.0,
    top_n=5
)
📊 Example Output
Top 5 recommended restaurants:
          name   cuisine  price  rating
  Bella Italia   Italian    300     4.5
   Roma DineIn   Italian    450     4.2
   Pasta House   Italian    350     4.1
📌 Future Improvements
- Add collaborative filtering for user-based recommendations
- Integrate with a web app (Flask/Django) for interactive use
- Support more advanced NLP for cuisine matching

👨‍💻 Author
Developed by Thiru.
Feel free to fork, contribute, or suggest improvements!
