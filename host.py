import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ----------------------------
# Define preprocessing (must match training!)
# ----------------------------
categorical_cols = ['batting_team', 'bowling_team', 'city']
numeric_cols = ['run_left', 'ball_left', 'wicket_left', 'total_runs_y', 'curr', 'req']

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# Dummy pipeline structure (classifier is placeholder, replaced by joblib.load)
dummy_model = RandomForestClassifier()
pipe_template = Pipeline(steps=[("preprocessor", preprocessor), ("model", dummy_model)])

# ----------------------------
# Load trained pipeline
# ----------------------------
try:
    pipe = joblib.load("pipe_compressed.pkl")
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    pipe = pipe_template  # fallback so app still runs

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="centered")

st.markdown("<h1 style='text-align: center; color: #FF5733;'>🏏 IPL Win Predictor 🏆</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# IPL Teams
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

# Cities where IPL is played
cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Two-column layout for team selection
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('🏏 Select the Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('🎯 Select the Bowling Team', sorted(teams))

if batting_team == bowling_team:
    st.warning("⚠️ Batting and Bowling teams cannot be the same. Please select different teams.")

# Host city selection
selected_city = st.selectbox('📍 Select Match Location', sorted(cities))

# Target Score Input
target = st.number_input('🎯 Target Score', min_value=0, max_value=500, value=0, step=1, format="%d")

# Three-column layout for match details
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('🏏 Current Score', min_value=0, max_value=500, step=1)

with col4:
    overs_options = [float(f"{over}.{ball}") for over in range(20) for ball in [0, 1, 2, 3, 4, 5]]
    overs = st.selectbox('⏳ Overs Completed', overs_options, index=0)

with col5:
    wickets = st.number_input('❌ Wickets Lost', min_value=0, max_value=10, value=0, step=1, format="%d")

# ----------------------------
# Prediction Button
# ----------------------------
if st.button('🔮 Predict Winning Probability'):

    if target == 0:
        st.error("❌ Please enter a valid target score.")
    elif overs == 0 and score > 0:
        st.error("❌ Overs cannot be 0 if a score is already set.")
    elif overs == 0:
        st.warning("⚠️ Please enter the number of overs to calculate probability.")
    elif batting_team == bowling_team:
        st.error("❌ Batting and Bowling teams cannot be the same.")
    else:
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [batting_team], 'bowling_team': [bowling_team],
            'city': [selected_city], 'run_left': [runs_left], 'ball_left': [balls_left],
            'wicket_left': [wickets_left], 'total_runs_y': [target], 'curr': [crr], 'req': [rrr]
        })

        try:
            result = pipe.predict_proba(input_df)
            loss_prob = result[0][0]
            win_prob = result[0][1]

            st.markdown(f"<h2 style='text-align: center; color: #0078FF;'>{batting_team} - {round(win_prob * 100)}% Chance to Win</h2>", unsafe_allow_html=True)
            st.progress(win_prob)

            st.markdown(f"<h2 style='text-align: center; color: #FF0000;'>{bowling_team} - {round(loss_prob * 100)}% Chance to Win</h2>", unsafe_allow_html=True)
            st.progress(loss_prob)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
