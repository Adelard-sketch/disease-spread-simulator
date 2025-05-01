# disease-spread-simulator
COVID-19 spread simulator using ML to predict infection risk from symptoms

Disease Spread Simulator is a machine learning-based web app developed using Streamlit to predict the likelihood of a COVID-19 infection based on user-entered symptoms. By analyzing symptoms like cough, fever, sore throat, and shortness of breath, along with demographic information such as age and gender, the app uses a trained machine learning model to assess the risk of infection. This tool provides real-time predictions and helps users understand their potential health risks, enabling more informed decisions regarding COVID-19 testing and precautions.


##Setup and Running Instructions

Install Dependencies: pip install -r requirements.txt
                      streamlit
                      scikit-learn
                      joblib
                      pandas
                      numpy
                      matplotlib
Requires Python 3.13.

Prepare Files:

Ensure covid_data_reduced.csv and covid_prediction_model.pkl are in the same directory as app.py.

Run the Streamlit App Locally: streamlit run app.py

Run the Notebook:

Open Virus_Spread_Similator_(1).ipynb in Google Colab or Jupyter Notebook. Update file paths (e.g., for covid_data_2020-2021.csv) as needed. Run all cells to preprocess data, train the model, and evaluate performance.

Alternative Interface:

Run Virus_Spread_Simulator_Implementation.py in Google Colab for an ipywidgets-based SIRD simulation interface.

Deployed Application The Virus Spread Simulator is deployed and accessible online at: https://disease-spread-simulator-k7v7etpkw5m3d4dlwnfmst.streamlit.app Use this link to interact with the app directly, input patient data for infection risk predictions, and explore SIRD simulation scenarios....
