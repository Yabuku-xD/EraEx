from src.api.main import app

if __name__ == '__main__':
    # Run the Flask app
    # Host='0.0.0.0' allows external access (like from your phone if on same WiFi)
    app.run(debug=True, port=5000, host='0.0.0.0')