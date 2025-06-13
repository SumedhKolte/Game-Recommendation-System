import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import requests


@st.cache_data
def load_data():
    # Replace with your hosted file URL
    URL = "https://drive.google.com/file/d/18Bvn9Hs7nEeDrxISBGIV_yfbIwMWCRhC/view?usp=drive_link/games.csv"
    try:
        df = pd.read_csv(URL)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


# --- Data Loading and Preprocessing ---
# Load your game dataset 
df = load_data()

# Clean data - remove rows with missing values in important columns
df = df.dropna(subset=['Name', 'Tags', 'Publishers', 'Developers', 'Genres', 'Categories'])

# Reset index after cleaning
df = df.reset_index(drop=True)

# Then proceed with the rest of the preprocessing
def combine_features(row):
    return ' '.join(str(row['Tags']).split(',')) + ' ' + \
           ' '.join(str(row['Publishers']).split(',')) + ' ' + \
           ' '.join(str(row['Developers']).split(',')) + ' ' + \
           ' '.join(str(row['Genres']).split(',')) + ' ' + \
           ' '.join(str(row['Categories']).split(','))
           
# Apply combine_features to create a new column
df['combined_features'] = df.apply(combine_features, axis=1)

# Create a TagSet column for Jaccard similarity
df['TagSet'] = df['Tags'].astype(str).apply(lambda x: set(x.split(',')))

# Create TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
tag_matrix = vectorizer.fit_transform(df['combined_features'])

# Get popular games based on 'Estimated owners'
popular_games = df.sort_values('Estimated owners', ascending=False)


# --- Recommendation Functions ---
def recommend_game_by_name(game_name, df, tag_matrix, n=10):
    try:
        if game_name not in df['Name'].values:
            return f"❌ Game '{game_name}' not found in dataset."

        game_index = df[df['Name'] == game_name].index[0]
        
        # Add validation for matrix size
        if game_index >= tag_matrix.shape[0]:
            return f"❌ Error: Game index out of range. Please try another game."
            
        similarities = cosine_similarity(tag_matrix[game_index], tag_matrix).flatten()
        top_indices = similarities.argsort()[-n-1:-1][::-1]
        recommendations = df.iloc[top_indices][['AppID', 'Name', 'Tags']].copy()
        recommendations['Similarity Score'] = similarities[top_indices]
        return recommendations.reset_index(drop=True)
    except Exception as e:
        return f"❌ Error processing game recommendations: {str(e)}"


def recommend_game_jaccard(game_name, df, n=10):
    if game_name not in df['Name'].values:
        return f"❌ Game '{game_name}' not found."
    target_tags = df[df['Name'] == game_name]['TagSet'].values[0]
    similarities = []
    for idx, row in df.iterrows():
        if row['Name'] == game_name:
            continue
        sim = len(target_tags.intersection(row['TagSet'])) / len(target_tags.union(row['TagSet']))
        similarities.append((row['AppID'], row['Name'], sim))
    similarities = sorted(similarities, key=lambda x: x[2], reverse=True)
    recommendations = similarities[:n]
    return pd.DataFrame(recommendations, columns=['AppID', 'Name', 'Jaccard Similarity Score'])


def recommend_similar_games(game_name, top_n=5):
    try:
        if game_name not in df['Name'].values:
            return []
        
        idx = df[df['Name'] == game_name].index[0]
        if idx >= tag_matrix.shape[0]:
            return []
            
        similarities = cosine_similarity(tag_matrix[idx], tag_matrix).flatten()
        sim_scores = list(enumerate(similarities))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_games = [df['Name'][i[0]] for i in sim_scores[1:top_n+1]]
        return top_games
    except Exception as e:
        return []

def hybrid_recommendation(game_name, top_n=5):
    try:
        content_recs = recommend_similar_games(game_name, top_n=50)
        if not content_recs:
            return []
            
        popularity_scores = {game: df[df['Name'] == game]['Estimated owners'].values[0] 
                           if df[df['Name'] == game]['Estimated owners'].values.size > 0 else 0 
                           for game in popular_games['Name']}
        ranked_recs = sorted(content_recs, key=lambda game: popularity_scores.get(game, 0), reverse=True)
        return ranked_recs[:top_n]
    except Exception as e:
        return []


def hybrid_recommendation(game_name, top_n=5):
    content_recs = recommend_similar_games(game_name, top_n=50)
    popularity_scores = {game: df[df['Name'] == game]['Estimated owners'].values[0] 
                          if df[df['Name'] == game]['Estimated owners'].values.size > 0 else 0 
                          for game in popular_games['Name']}
    ranked_recs = sorted(content_recs, key=lambda game: popularity_scores.get(game, 0), reverse=True)
    return ranked_recs[:top_n]


# --- Fuzzy Search Function ---
def search_similar_game_names(user_input, df, limit=10):
    game_names = df['Name'].dropna().tolist()
    matches = difflib.get_close_matches(user_input, game_names, n=limit, cutoff=0.4)
    return matches if matches else ["❌ No similar game names found."]


# --- Streamlit UI ---
st.set_page_config(page_title="Game Recommender", page_icon="🎮", layout="centered")
st.title("🎮 Game Recommendation System")

# User Input
user_input = st.text_input("🔍 Search for a game:", "")

if user_input:
    # Similar Game Names
    suggestions = search_similar_game_names(user_input, df, limit=5)
    if "❌" in suggestions[0]:
        st.error(suggestions[0])
    else:
        selected_game = st.selectbox("Choose the correct game:", suggestions)

        if st.button("Get Recommendations"):
            with st.spinner("🔍 Finding similar games..."):
                # Cosine Similarity Recommendations
                cos_recs = recommend_game_by_name(selected_game, df, tag_matrix, n=5)
                if isinstance(cos_recs, str):
                    st.error(cos_recs)
                else:
                    st.subheader("🎯 Recommendations (Cosine Similarity)")
                    st.dataframe(cos_recs)

                # Jaccard Similarity Recommendations
                jac_recs = recommend_game_jaccard(selected_game, df, n=5)
                if isinstance(jac_recs, str):
                    st.error(jac_recs)
                else:
                    st.subheader("🎯 Recommendations (Jaccard Similarity)")
                    st.dataframe(jac_recs)

                # Hybrid Recommendations
                hybrid_recs = hybrid_recommendation(selected_game, top_n=5)
                st.subheader("🎯 Hybrid Recommendations (Popularity-based)")
                st.dataframe(pd.DataFrame(hybrid_recs, columns=['Name']))