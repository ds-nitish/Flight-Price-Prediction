from flask import Flask
import pickle
from flask import request , render_template ,jsonify
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
le_airline = pickle.load(open('le_airline.pkl','rb'))
le_arrival_time = pickle.load(open('le_arrival_time.pkl','rb'))
le_class = pickle.load(open('le_class.pkl','rb'))
le_departure_time = pickle.load(open('le_departure_time.pkl','rb'))
le_destination_city = pickle.load(open('le_destination_city.pkl','rb'))
le_source_city = pickle.load(open('le_source_city.pkl','rb'))
le_stops = pickle.load(open('le_stops.pkl','rb'))

@app.route("/")
def loadpage():
    return render_template('home.html')


@app.route('/predict' , methods = ['post'])
def prediction():

    Airline_Name = request.form['Airline Name']
    Source_City = request.form['Source City']
    Departure_Time = request.form['Departure Time']
    Stop = request.form['Stop']
    Arrival_Time = request.form['Arrival Time']
    Destination_City = request.form['Destination City']
    Class = request.form['Class']
    Days_Left = request.form['Days Left']

    ip = np.array([[Airline_Name,Source_City,Departure_Time,Stop,Arrival_Time,Destination_City,Class,Days_Left]])

    ip[:,0] = le_airline.transform(ip[:,0]) 
    ip[:,1] = le_source_city.transform(ip[:,1])
    ip[:,2] = le_departure_time.transform(ip[:,2])
    ip[:,3] = le_stops.transform(ip[:,3])
    ip[:,4] = le_arrival_time.transform(ip[:,4])
    ip[:,5] = le_destination_city.transform(ip[:,5])
    ip[:,6] = le_class.transform(ip[:,6])
    ip = ip.astype(int)

    pred = model.predict(ip)
    output = round(pred[0],2)
    print(output)
    
    return render_template('home.html',prediction_text = 'Predicted Price is Rs {}'.format(output))



if __name__ == '__main__':
    app.run(debug = True)
