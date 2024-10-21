from flask import Flask, request, jsonify
import sqlite3
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')  # Ensure the correct path

def get_location_data(location):
    # Connect to SQLite and fetch average speed and congestion
    conn = sqlite3.connect('B_dataset.db')
    cursor = conn.cursor()

    query = '''SELECT average_speed, congestion FROM traffic_data WHERE location = ?'''
    cursor.execute(query, (location,))
    data = cursor.fetchone()

    conn.close()

    if data:
        return {'average_speed': data[0], 'congestion': data[1]}
    else:
        return None

@app.route('/predict', methods=['POST'])
def predict_cycle_time():
    request_data = request.get_json()
    location = request_data.get('location')

    # Fetch data from SQLite database
    location_data = get_location_data(location)

    if not location_data:
        return jsonify({'error': 'Location data not found'}), 404

    # Prepare feature array for model prediction
    features = [location_data['average_speed'], location_data['congestion']]

    # Predict cycle time using the model
    cycle_time = model.predict([features])[0]  # Ensure the model is trained accordingly

    return jsonify({'cycleTime': cycle_time})

if __name__ == '__main__':
    app.run(debug=True)
