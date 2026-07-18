from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
import datetime as dt

import numpy as np
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ===================== ML MODEL: SIMPLE SYNTHETIC TRAINING =====================

def make_synthetic_data():
    """
    Hum yaha fake data bana rahe hain:
    features: [hour(0-23), is_weekend(0/1)]
    target : traffic volume (0-100)
    """
    X = []
    y = []

    for hour in range(24):
        for is_weekend in [0, 1]:
            # base volume rules:
            if (7 <= hour <= 11) or (17 <= hour <= 21):
                base = 90  # rush
            elif 11 <= hour < 17:
                base = 60  # din me medium
            else:
                base = 15  # raat / subah

            # weekend me thoda kam
            if is_weekend:
                base -= 10

            noise = np.random.normal(0, 5)  # random thoda sa
            X.append([hour, is_weekend])
            y.append(max(5, min(100, base + noise)))

    return np.array(X), np.array(y)

X, y = make_synthetic_data()
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)


# ===================== DATABASE (SQLite) =====================

DB_PATH = "traffic.db"

def get_db():
    make = not os.path.exists(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    if make:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lat REAL,
                lon REAL,
                hour INTEGER,
                volume REAL,
                level TEXT,
                created_at TEXT
            )
            """
        )
        conn.commit()
    return conn


def volume_to_level(v):
    if v >= 75:
        return "HIGH"
    elif v >= 35:
        return "MEDIUM"
    else:
        return "LOW"


# ===================== ROUTES =====================

@app.route("/")
def index():
    # static/index.html serve karega
    return app.send_static_file("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Body JSON:
    {
      "lat": 26.8,
      "lon": 80.9,
      "hour": 14   // optional, nahi bhejo toh current hour use hoga
    }
    """
    data = request.get_json() or {}
    lat = float(data.get("lat", 0))
    lon = float(data.get("lon", 0))

    now = dt.datetime.now()
    hour = int(data.get("hour", now.hour))
    is_weekend = 1 if now.weekday() >= 5 else 0

    volume = float(model.predict([[hour, is_weekend]])[0])
    level = volume_to_level(volume)

    conn = get_db()
    conn.execute(
        "INSERT INTO records (lat, lon, hour, volume, level, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (lat, lon, hour, volume, level, dt.datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()

    return jsonify(
        {
            "lat": lat,
            "lon": lon,
            "hour": hour,
            "is_weekend": bool(is_weekend),
            "volume": volume,
            "level": level,
        }
    )


@app.route("/api/daily-curve", methods=["GET"])
def api_daily_curve():
    """
    Query params:
      ?lat=..&lon=..
    Response: 24 hours ka curve (0-23)
    """
    lat = float(request.args.get("lat", 0))
    lon = float(request.args.get("lon", 0))

    now = dt.datetime.now()
    is_weekend = 1 if now.weekday() >= 5 else 0

    hours = list(range(24))
    volumes = []
    levels = []

    for h in hours:
        v = float(model.predict([[h, is_weekend]])[0])
        volumes.append(v)
        levels.append(volume_to_level(v))

    return jsonify(
        {
            "lat": lat,
            "lon": lon,
            "is_weekend": bool(is_weekend),
            "hours": hours,
            "volumes": volumes,
            "levels": levels,
        }
    )


if __name__ == "__main__":
    # debug = True rakho development me, project dikhane ke liye bhi chalega
    app.run(host="0.0.0.0", port=5000, debug=True)
