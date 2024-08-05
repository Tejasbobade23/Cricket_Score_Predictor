import streamlit as st
import pickle
import pandas as pd
import numpy as np


pipe = pickle.load(open('pipe.pkl','rb'))

teams = ['Australia', 'New Zealand', 'South Africa', 'England', 
'India', 'West Indies', 'Pakistan', 'Bangladesh', 'Afghanistan', 'Sri Lanka']

cities = ['Colombo','Mirpur','Johannesburg','Dubai','Auckland','Cape Town',
'London','Pallekele','Barbados','Sydney','Melbourne',
'Durban','St Lucia','Wellington','Lauderhill','Hamilton',
'Centurion','Manchester','Abu Dhabi','Mumbai','Nottingham','Southampton',
'Mount Maunganui','Chittagong','Kolkata','Lahore','Delhi',
'Nagpur','Chandigarh','Adelaide','Bangalore','St Kitts','Cardiff',
'Christchurch','Trinidad']


st.title('Cricket Score Predictor')

col1, col2 = st.columns(2)

# Select batting team
with col1:
    batting_team = st.selectbox('Select batting team', sorted(teams))

# Select bowling team
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

# To select city
city = st.selectbox('Select city', sorted(cities))

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score')

with col4:
    overs_completed = st.number_input('Overs completed(works for over > 5)')

with col5:
    wickets = st.number_input('Wickets Out')

last_five = st.number_input('Runs scored in last 5 overs')

if st.button('Predict Score'):
    balls_left = 120 - (overs_completed * 6)
    wicket_left = 10 - wickets
    current_run_rate = current_score/overs_completed

    input_df = pd.DataFrame(
        {'batting_team' : [batting_team], 'bowling_team' : [bowling_team], 'city' : city, 'current_score' : [current_score],
    'balls_left' : [balls_left], 'wicket_left' : [wickets], 'current_run_rate' : [current_run_rate],
    'last_five' : [last_five]})

    # st.text(xgboost.__version__)
    result = pipe.predict(input_df)
    st.header("Predicted Score: " + str(int(result[0])))