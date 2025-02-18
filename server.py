from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/location', methods=['POST'])
def receive_location():
    data = request.json
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    print(f"Received Location: Latitude={latitude}, Longitude={longitude}")
    return jsonify({"status": "success", "latitude": latitude, "longitude": longitude})

if __name__ == '__main__':
    app.run(debug=True)

