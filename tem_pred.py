# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 14:52:27 2023

@author: Benna
"""

import streamlit as st
import pickle 
import pandas as pd
import datetime


#loading the model
loaded_model=pickle.load(open("C:/Users/Benna/Trained_prophet.sav",'rb'))


import plotly.express as px




def temperature_predictions(date_start, date_end, loaded_model):
   future_dates = pd.date_range(start=date_start, end=date_end, freq='M') 
   #future_dataframe = loaded_model.make_future_dataframe(periods=len(future_dates+pd.DateOffset(years=10) ),freq='M', include_history=False)
   #future_dataframe = loaded_model.make_future_dataframe(periods=len(future_dates),freq='M', include_history=False)
   future_dataframe = loaded_model.make_future_dataframe(periods= 120 + len(future_dates),freq='M', include_history=False)
   
   #print(loaded_model.date_end)

   #print(future_dataframe)
   forecast = loaded_model.predict(future_dataframe)
    
   forecast['yhat'] = forecast['yhat'] + 10
   #forecast['ds'] = forecast['ds'] + pd.DateOffset(years=10)
   today = pd.Timestamp(datetime.date.today())

   filtered_forecast = forecast.loc[(forecast['ds'] >= today)]
   filtered_forecast.rename(columns = {'yhat':'Upper forecast', 'yhat_upper':'Forecast', 'yhat_lower':'Lower forecast'}, inplace=True )
   print(filtered_forecast.columns)
   fig = px.line(filtered_forecast, x='ds', y=['Upper forecast', 'Forecast', 'Lower forecast'], title='Average land temperature in Tunisia', color_discrete_sequence=["red", "green", "blue"])
   fig.show()





    

def main(): 
    st.title("Temperature Prediction Web App")
    st.write("Hello 👋")

    date_start=st.date_input("The starting date")
    date_end=st.date_input("The Ending date")

    
    
    
    
    result=""
        
    if st.button("Predict"):
        result= temperature_predictions(date_start, date_end,loaded_model)
        st.success(result)

    if st.button("More Info"):
        st.write("Temperature forecasting using facebook prophet ")
        st.write("Powered by Abdelmomen")
if __name__=='__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    