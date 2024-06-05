from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load the trained model
with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extracting data from the form
        date = request.form['date']
        hour = int(request.form['hour'])
        junction = int(request.form['junction'])
        day_of_week = request.form['day_of_week']
        
        # Parse the date
        year, month, day = map(int, date.split('-'))

        # Convert day_of_week to one-hot encoding
        day_of_week_features = ['day_of_week_Friday', 'day_of_week_Monday', 'day_of_week_Saturday',
                                'day_of_week_Sunday', 'day_of_week_Thursday', 'day_of_week_Tuesday', 'day_of_week_Wednesday']
        day_of_week_data = np.zeros(len(day_of_week_features))
        day_of_week_index = day_of_week_features.index(f'day_of_week_{day_of_week}')
        day_of_week_data[day_of_week_index] = 1

        # Creating the input array
        input_data = np.array([year, month, day, hour, junction] + list(day_of_week_data)).reshape(1, -1)

        # Making prediction
        prediction = rf_model.predict(input_data)[0]

        return render_template('result.html', prediction=int(prediction))

if __name__ == "__main__":
    app.run(debug=True)
