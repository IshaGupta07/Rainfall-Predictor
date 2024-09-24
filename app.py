from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import calendar

# Load the trained model
model = joblib.load('rainfall_model.pkl')
data = pd.read_csv("dataset.csv")

# Get unique states for the dropdown menu
states = data['SUBDIVISION'].unique()

# Create a dictionary for months
months = {i: calendar.month_name[i] for i in range(1, 13)}

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        state = request.form['state']
        year = int(request.form['year'])
        month = int(request.form['month'])
        
        # Process data for the selected state
        group = data.groupby('SUBDIVISION')[['YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']]
        state_data = group.get_group(state)
        
        # Reshape the data for modeling
        df = state_data.melt(['YEAR']).reset_index()
        df = df[['YEAR', 'variable', 'value']].reset_index().sort_values(by=['YEAR', 'index'])
        df.columns = ['Index', 'Year', 'Month', 'Avg_Rainfall']
        
        # Prepare features for prediction
        X = np.array([[year, month]])
        
        # Predict
        predicted_rainfall = model.predict(X)[0]
        
        # Get month name from month number
        month_name = calendar.month_name[month]
        
        return render_template('index.html', prediction=predicted_rainfall, state=state, year=year, month_name=month_name, states=states, months=months)
    
    return render_template('index.html', prediction=None, states=states, months=months)

if __name__ == '__main__':
    app.run(debug=True)