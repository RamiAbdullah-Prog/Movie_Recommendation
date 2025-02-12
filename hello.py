import streamlit as st  
import pandas as pd 
from sentence_transformers import SentenceTransformer , util

st.title('Movie Recommender')
# streamlit run hello.py

st.write('This is a movie recommendation app using Streamlit and Python')

data = pd.read_csv('movies.csv')


movie = st.text_input('Enter your favourite movie', key='movie')

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_search(movie_title, data):
    
    movie_titles = data['movie_name'].tolist()

    title_embeddings = model.encode(movie_title, convert_to_tensor=True)
    movie_embeddings = model.encode(movie_titles, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(title_embeddings, movie_embeddings)[0]

    best_similar = similarity.argmax().item()

    best_similar_title = movie_titles[best_similar]
    best_similar_score = data.iloc[best_similar]['imdb_rating']
    
    return best_similar_title, best_similar_score


if st.button('Movie IMDB Score !'):
    
    if movie :
       
        movie_title , recommendation = semantic_search(movie, data)
        st.write(recommendation)
        st.write( movie_title)

st.write (data)
