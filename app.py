import streamlit as st
import streamlit.components.v1 as stc
import pickle
import pandas as pd

try:
    with open('best_poly_model.pkl', 'rb') as file:
        best_poly_model = pickle.load(file)
    with open ('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.error("Error: 'scaler.pkl' or 'best_lasso_model.pkl' not found.")
    st.stop()
except pickle.PickleError:
    st.error("Error: Failed to load 'scaler.pkl' or 'best_poly_model.pkl'. The files might be corrupted.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

html_temp = """<div style="background-color:#000;padding:10px;border-radius:10px">
                <h1 style="color:#fff;text-align:center">Song Popularity Prediction App</h1> 
                <h4 style="color:#fff;text-align:center">Made for: Credit Team</h4> 
                """

desc_temp = """ ### Song Popularity Prediction App 
                This app is used by Cakrawala Team for predicting Song Popularity
                
                #### Data Source
                Kaggle: https://www.kaggle.com/datasets/yasserh/song-popularity-dataset
                """

def main():
    stc.html(html_temp)
    menu = ["Home", "Machine Learning App"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Machine Learning App":
        run_ml_app()

# One-Hot Encoding
def one_hot_encode_time_signature(selected_ts):
    encoded = {}
    valid_time_signatures = [1, 3, 4, 5]
    for ts in valid_time_signatures:
        encoded[f'time_signature_{ts}'] = [1 if selected_ts == ts else 0]
    return encoded

def one_hot_encode_key(selected_key):
    encoded = {}
    for i in range(1,12):
        encoded[f'key_{i}'] = [1 if selected_key == i else 0]
    return encoded

def run_ml_app():
    design = """<div style="padding:15px;">
                    <h1 style="color:#fff">Song Popularity Prediction</h1>
                </div
             """
    
    st.markdown(design, unsafe_allow_html=True)
    left,right = st.columns((2,2))
    song_duration_ms = left.number_input('Song Duration (ms)', min_value=0, max_value=600000, step=1000)
    acousticness = right.number_input('Acousticness', min_value=0.00, max_value=1.00, step=0.01)
    danceability = left.number_input('Danceability', min_value=0.00, max_value=1.00, step=0.01)
    energy = right.number_input('Energy', min_value=0.00, max_value=1.00, step=0.01)
    instrumentalness = left.number_input('Instrumentalness', min_value=0.00, max_value=1.00, step=0.01)
    liveness = right.number_input('Liveness', min_value=0.00, max_value=1.00, step=0.01)
    loudness = left.number_input('Loudness', min_value=-60.00, max_value=0.00, step=0.01)
    audio_mode = right.selectbox('Audio Mode', [0,1])
    speechiness = left.number_input('Speechiness', min_value=0.00, max_value=1.00, step=0.01)
    tempo = right.number_input('Tempo', min_value=0.00, max_value=1.00, step=0.01)
    audio_valence = left.number_input('Audio Valence', min_value=0.00, max_value=1.00, step=0.01)
    selected_ts = right.selectbox('Time Signature', [1,3,4,5])
    selected_key = st.selectbox('Key', list(range(1,12)))
    button = st.button('Predict')

    time_signature_encoded = one_hot_encode_time_signature(selected_ts)
    key_encoded = one_hot_encode_key(selected_key)
    
    # Making Dictionary
    features = {
        'song_duration_ms':[song_duration_ms],
        'acousticness':[acousticness],
        'danceability':[danceability],
        'energy':[energy],
        'instrumentalness':[instrumentalness],
        'liveness':[liveness],
        'loudness':[loudness],
        'audio_mode':[audio_mode],
        'speechiness':[speechiness],
        'tempo':[tempo],
        'audio_valence':[audio_valence],
        **time_signature_encoded,
        **key_encoded
    }

    data = pd.DataFrame(features, columns=[
        'song_duration_ms', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness',
        'audio_mode', 'speechiness', 'tempo', 'audio_valence', 'time_signature_1', 'time_signature_3', 'time_signature_4',
        'time_signature_5', 'key_1', 'key_2', 'key_3', 'key_4', 'key_5', 'key_6', 'key_7', 'key_8', 'key_9', 'key_10', 'key_11'
    ])
        
    # If button is clilcked
    if button:
        if data.shape[1] == 9:
            # Transformation with scaler
            data_scaled = scaler.transform(data)
            
            # Making prediction
            prediction = best_poly_model.predict(data_scaled)
            st.success(f'Predicted Song Popularity: {prediction[0]:.2f}')
        else:
            st.error(f'Error: Incorrect number of features. Expected 26, but got {data.shape[1]}.')

if __name__ == "__main__":
    main()
