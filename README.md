# IPL-Match-Predictor
IPL Match Predictor
This project predicts the outcome of IPL matches using machine learning models. The model is trained using two datasets: deliveries (de.csv) and matches (matches.csv).

Project Structure
de.csv – Ball-by-ball delivery dataset
matches.csv – Match-level dataset
creating_model.ipynb – Contains data preprocessing, EDA, and model training (Logistic Regression & Random Forest)
pipe.zip – Saved model pipeline
host.py – Streamlit-based UI for predictions
Model Selection
Both Logistic Regression and Random Forest were tested. Random Forest performed better and is used in the final model.

How to Run
Install dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run host.py


follow this https://ipl-match-prediction.streamlit.app/
