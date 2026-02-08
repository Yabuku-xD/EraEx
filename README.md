# EraEx - The 2012-2018 Nostalgia Machine

A retro-styled music discovery engine that takes you back to the golden era of streaming.

## 🚀 How to Run

### 1. Setup
Install requirements:
```bash
pip install -r requirements.txt
```

Create a `.env` file with your Last.fm API key (already done).

### 2. Start the App
```bash
python src/api/main.py
```
**Open your browser to:** [http://localhost:5000](http://localhost:5000)

## ✨ Features
- **Nostalgia Search**: Finds tracks STRICTLY from 2012-2018.
- **Mood Filter**: Filter by "Hype", "Sad", "Chill", etc.
- **Audio Previews**: Listen to 30s clips directly in the browser.
- **Retro UI**: Frutiger Aero / Vaporwave aesthetic.

## 💻 Tech Stack
- **Backend**: Flask, Deezer API (Search), Last.fm API (User Profiles)
- **Frontend**: HTML5, CSS3 (Glassmorphism), Vanilla JS
- **Models**: ALS (for personalized recommendations, optional)
