from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
with open('Final_Model.pkl', 'rb') as f:
    model = pickle.load(f)

# Encoders based on your dataset
manufacturer_map = {
    'AUDI': 0, 'BENTLEY': 1, 'BMW': 2, 'CHEVROLET': 3, 'DODGE': 4, 'FERRARI': 5,
    'FIAT': 6, 'FORD': 7, 'GMC': 8, 'HONDA': 9, 'HYUNDAI': 10, 'ISUZU': 11,
    'JAGUAR': 12, 'JEEP': 13, 'KIA': 14, 'LAMBORGHINI': 15, 'LANCIA': 16,
    'LAND ROVER': 17, 'MERCEDES-BENZ': 18, 'NISSAN': 19, 'PORSCHE': 20,
    'RENAULT': 21, 'ROLLS-ROYCE': 22, 'SKODA': 23, 'SUZUKI': 24, 'TOYOTA': 25,
    'UAZ': 26, 'VAZ': 27, 'VOLKSWAGEN': 28, 'VOLVO': 29
}

fuel_map = {
    'Petrol': 0, 'Hybrid': 1, 'Diesel': 2, 'CNG': 3, 'LPG': 4,
    'Plug-in Hybrid': 5, 'Hydrogen': 6
}

gearbox_map = {
    'Automatic': 0, 'Tiptronic': 1, 'Manual': 2, 'Variator': 3
}

@app.route('/', methods=['GET'])
def home():
    return render_template('fend.html', price=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        manufacturer = request.form['manufacturer'].upper()
        prod_year = int(request.form['prodYear'])
        leather = 1 if request.form['leatherInterior'] == 'Yes' else 0
        fuel_type = request.form['fuelType']
        engine_vol = float(request.form['engineVolume'])
        gearbox = request.form['gearboxType']
        airbags = int(request.form['airbags'])

        # Encode categorical values
        manufacturer_encoded = manufacturer_map.get(manufacturer, -1)
        fuel_encoded = fuel_map.get(fuel_type, -1)
        gearbox_encoded = gearbox_map.get(gearbox, -1)

        if -1 in [manufacturer_encoded, fuel_encoded, gearbox_encoded]:
            raise ValueError("Invalid categorical value")

        # Match training feature names
        input_df = pd.DataFrame([[manufacturer_encoded, prod_year, leather, fuel_encoded, engine_vol, gearbox_encoded, airbags]],
            columns=['Manufacturer', 'Prod. year', 'Leather interior', 'Fuel type', 'Engine volume', 'Gear box type', 'Airbags'])

        prediction = model.predict(input_df)[0]
        return render_template('fend.html', price=round(prediction, 2))

    except Exception as e:
        return render_template('fend.html', price=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
