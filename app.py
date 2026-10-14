import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import requests

import predict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE = "https://nominatim.openstreetmap.org/reverse"
BIGDATACLOUD_REVERSE = "https://api.bigdatacloud.net/data/reverse-geocode-client"
UA = {"User-Agent": "TrafficVision/1.0 (student project demo)"}

 
@app.route("/")
def home():
    now = datetime.now()
    return render_template(
        "index.html",
        current_date=now.strftime("%d %B %Y"),
        current_time=now.strftime("%I:%M %p")
    )


@app.route("/api/geocode")
def geocode():
    """Place search. Nominatim first (knows localities like Gachibowli that
    Open-Meteo lacks), Open-Meteo as fallback. Failures are logged to the
    terminal so network problems can be diagnosed."""
    query = request.args.get("q", "").strip()
    if len(query) < 2:
        return jsonify([])
    try:
        response = requests.get(
            NOMINATIM_SEARCH,
            params={"q": query, "format": "json", "limit": 8},
            headers=UA,
            timeout=8,
        )
        response.raise_for_status()
        results = [
            {"name": item["display_name"],
             "lat": float(item["lat"]), "lon": float(item["lon"])}
            for item in response.json()
        ]
        if results:
            return jsonify(results)
    except requests.RequestException as exc:
        print(f"[geocode] nominatim failed for '{query}': "
              f"{type(exc).__name__}: {exc}", flush=True)
    try:
        response = requests.get(
            GEOCODE_URL,
            params={"name": query, "count": 8, "language": "en", "format": "json"},
            timeout=10
        )
        response.raise_for_status()
        payload = response.json()
        results = []
        for item in payload.get("results", []):
            parts = [item.get("name"), item.get("admin1"), item.get("country")]
            label = ", ".join(dict.fromkeys(p for p in parts if p))
            results.append({
                "name": label,
                "lat": float(item["latitude"]),
                "lon": float(item["longitude"])
            })
        return jsonify(results)
    except requests.RequestException as exc:
        print(f"[geocode] open-meteo fallback failed for '{query}': "
              f"{type(exc).__name__}: {exc}", flush=True)
        return jsonify({"error": "Location search service is unavailable. Check your internet connection."}), 502


@app.route("/api/reverse")
def reverse_geocode():
    """GPS coords -> a readable place name like 'Gachibowli, Hyderabad'.

    Nominatim first (clean area/city names from address parts), then
    Open-Meteo, then BigDataCloud. Failures are logged to the terminal.
    Returns 502 if no service can resolve the coordinates."""
    try:
        lat = float(request.args["lat"])
        lon = float(request.args["lon"])
    except (TypeError, ValueError, KeyError):
        return jsonify({"error": "lat and lon are required."}), 400
    try:
        response = requests.get(
            NOMINATIM_REVERSE,
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 14},
            headers=UA,
            timeout=8,
        )
        response.raise_for_status()
        addr = (response.json() or {}).get("address", {})
        area = (addr.get("suburb") or addr.get("neighbourhood")
                or addr.get("quarter") or addr.get("hamlet"))
        city = (addr.get("city") or addr.get("town")
                or addr.get("village") or addr.get("county"))
        place = ", ".join(p for p in [area, city] if p)
        if place:
            return jsonify({"place": place, "lat": lat, "lon": lon})
    except requests.RequestException as exc:
        print(f"[reverse] nominatim failed for ({lat}, {lon}): "
              f"{type(exc).__name__}: {exc}", flush=True)
    try:
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/reverse",
            params={"latitude": lat, "longitude": lon, "language": "en", "format": "json"},
            timeout=10
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if results:
            item = results[0]
            parts = [item.get("name"), item.get("admin1"), item.get("country")]
            place = ", ".join(dict.fromkeys(p for p in parts if p))
            if place:
                return jsonify({"place": place, "lat": lat, "lon": lon})
    except requests.RequestException as exc:
        print(f"[reverse] open-meteo failed for ({lat}, {lon}): "
              f"{type(exc).__name__}: {exc}", flush=True)
    try:
        response = requests.get(
            BIGDATACLOUD_REVERSE,
            params={"latitude": lat, "longitude": lon, "localityLanguage": "en"},
            timeout=8,
        )
        response.raise_for_status()
        j = response.json() or {}
        place = ", ".join(p for p in
                          [j.get("city") or j.get("locality"),
                           j.get("principalSubdivision")] if p)
        if place:
            return jsonify({"place": place, "lat": lat, "lon": lon})
    except requests.RequestException as exc:
        print(f"[reverse] bigdatacloud failed for ({lat}, {lon}): "
              f"{type(exc).__name__}: {exc}", flush=True)
    return jsonify({"error": "Could not resolve a place name for these coordinates."}), 502


@app.route("/api/weather")
def weather():
    try:
        lat = float(request.args["lat"])
        lon = float(request.args["lon"])
        response = requests.get(
            WEATHER_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,rain,showers,snowfall,cloud_cover,weather_code,wind_speed_10m",
                "hourly": "temperature_2m,precipitation_probability,precipitation,weather_code",
                "forecast_days": 2,
                "timezone": "auto"
            },
            timeout=12
        )
        response.raise_for_status()
        data = response.json()
        current = data.get("current", {})
        code = int(current.get("weather_code", 0))
        return jsonify({
            "live": True,
            "temp_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_kmh": current.get("wind_speed_10m"),
            "rain_mm": current.get("rain", current.get("precipitation", 0)),
            "raining": float(current.get("rain", 0) or 0) > 0 or float(current.get("precipitation", 0) or 0) > 0,
            "name": weather_description(code),
            "timezone": data.get("timezone", "local time"),
            "hourly": data.get("hourly", {})
        })
    except (ValueError, KeyError, requests.RequestException):
        return jsonify({"live": False, "error": "Live weather is unavailable right now."}), 200


def weather_description(code):
    if code == 0:
        return "Clear sky"
    if code in (1, 2, 3):
        return "Mainly clear / cloudy"
    if code in (45, 48):
        return "Fog"
    if code in (51, 53, 55, 56, 57):
        return "Drizzle"
    if code in (61, 63, 65, 66, 67):
        return "Rain"
    if code in (71, 73, 75, 77):
        return "Snow"
    if code in (80, 81, 82):
        return "Rain showers"
    if code in (85, 86):
        return "Snow showers"
    if code in (95, 96, 99):
        return "Thunderstorm"
    return "Unknown conditions"


@app.route("/predict", methods=["POST"])
def predict_route():
    body = request.get_json(silent=True) or request.form
    if body.get("lat") is None or body.get("lon") is None:
        return jsonify({"error": "No location selected. Search for a place or use GPS first."}), 400
    try:
        lat = float(body.get("lat"))
        lon = float(body.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid coordinates."}), 400

    place = str(body.get("place", "Selected location"))[:200]
    holiday = bool(body.get("holiday", False))
    festival = bool(body.get("festival", False))
    festival_name = str(body.get("festival_name", "")).strip()[:100]
    road_type = str(body.get("road_type", "highway")).lower()

    try:
        weather_data = requests.get(
            WEATHER_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,precipitation,rain,weather_code",
                "hourly": "temperature_2m,precipitation_probability,precipitation,weather_code",
                "forecast_days": 2,
                "timezone": "auto"
            },
            timeout=12
        ).json()
    except Exception:
        weather_data = {}

    # Forecast is based on the historical training dataset. GPS is used for place
    # selection and weather, not to magically create local traffic-sensor history.
    try:
        result = predict.forecast_next_24_hours(
            holiday=holiday or festival,
            festival=festival,
            festival_name=festival_name,
            weather_payload=weather_data
        )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        app.logger.exception("Forecast failed")
        return jsonify({"error": f"Forecast failed: {exc}"}), 500

    rows = result["hourly"]
    best = min(rows, key=lambda row: row["volume"])
    return jsonify({
        "place": place,
        "latitude": lat,
        "longitude": lon,
        "as_of": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "source_note": "Forecast model trained on the historical Metro Interstate I-94 dataset (US, through 2018). It is not verified Hyderabad traffic data.",
        "best_time": best,
        "labels": [r["label"] for r in rows],
        "hourly": [r["volume"] for r in rows],
        "rows": rows,
        "weather": {
            "live": bool(weather_data.get("current")),
            "temp_c": weather_data.get("current", {}).get("temperature_2m"),
            "name": weather_description(int(weather_data.get("current", {}).get("weather_code", 0))),
            "raining": float(weather_data.get("current", {}).get("precipitation", 0) or 0) > 0
        }
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
