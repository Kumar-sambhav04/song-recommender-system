import streamlit as st
import pandas as pd
import pickle
import requests


st.set_page_config(
    page_title="Spotify Song Recommender",
    #page_icon="🎵",
    #layout="wide"
)
st.image("images/banner1.png", width=1000)
st.header("Song Recommender System")
st.write("Find song similar to your Request")

st.sidebar.title("About")
st.sidebar.write(
    "This is a Machine Learning based Spotify song recommendation system that suggest similar song based on user selection.   "
    "The project uses KNN and Count Vectorization to find similarities and generate recommendations.\n " 
    "The suggesions are based on the data set used to make this project, so it may not have all the songs that are available on spotify")

def fetch_poster(song_name, artist_name):
    query = f"{song_name} {artist_name}"
    url = f"https://itunes.apple.com/search?term={query}&limit=1"

    response = requests.get(url)
    data = response.json()

    if data["resultCount"] > 0:
        return data["results"][0]["artworkUrl100"]

    else:
        return "https://via.placeholder.com/300"

new_df = pickle.load(open("song_.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))
vectors = pickle.load(open("vectors.pkl","rb"))

selected_song = st.selectbox(
    "Select a Song",
    new_df["track_name"].values,
    index= None,
    placeholder= "Enter Song",
)

def song_re(song):

    song_index = new_df[(new_df["track_name"] == song) | (new_df["artists"] == song)].index[0]

    distances,indices =  model.kneighbors(
        vectors[song_index:song_index+1],
        n_neighbors = 6
    )
    rec_songs = []
    for el in indices[0]:
        song_name = new_df.loc[el]["track_name"]
        song_track_id = new_df.loc[el]["track_id"]
        artist_name = new_df.iloc[el]["artists"]

        spotify_link = f"https://open.spotify.com/track/{song_track_id}"

        poster = fetch_poster(song_name,artist_name)

        rec_songs.append((song_name, spotify_link, poster))

    return rec_songs


if st.button("Recommend"):
    #st.write(selected_song)
    st.subheader("Recommended Song")


    cols = st.columns(3)
    with st.spinner("Finding recommendations..."):
        recommendations = song_re(selected_song)
    st.success("Done!")
    for idx, (song_name,spotify_link,poster) in enumerate(recommendations):
        with cols[idx % 3]:
            st.image(poster)
            st.markdown(f"[{song_name}]({spotify_link})")
