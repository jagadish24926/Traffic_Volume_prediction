from flask import Flask, render_template, request
import subprocess
import sys
import os
import re
from datetime import datetime

app = Flask(__name__)


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    now = datetime.now()

    current_date = now.strftime("%d %B %Y")
    current_time = now.strftime("%I:%M:%S %p")

    return render_template(
        "index.html",
        current_date=current_date,
        current_time=current_time
    )


# ==========================================
# PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------------
    # GET USER INPUT
    # --------------------------------------

    location = request.form.get("location", "").strip()

    latitude = request.form.get("latitude", "").strip()

    longitude = request.form.get("longitude", "").strip()

    holiday = request.form.get("holiday", "No")


    # --------------------------------------
    # CHECK LOCATION
    # --------------------------------------

    if not location:

        return render_template(
            "index.html",
            current_date=datetime.now().strftime("%d %B %Y"),
            current_time=datetime.now().strftime("%I:%M:%S %p"),
            error="Please enter a location."
        )


    # --------------------------------------
    # DISPLAY RECEIVED LOCATION
    # --------------------------------------

    print()
    print("===================================")
    print("LOCATION RECEIVED")
    print("===================================")

    print("Location  :", location)
    print("Latitude  :", latitude)
    print("Longitude :", longitude)
    print("Holiday   :", holiday)

    print("===================================")


    # --------------------------------------
    # RUN CURRENT TFT PREDICTION
    # --------------------------------------

    try:

        result = subprocess.run(
            [
                sys.executable,
                "predict.py"
            ],

            capture_output=True,

            text=True,

            cwd=os.path.dirname(
                os.path.abspath(__file__)
            )
        )


        # ----------------------------------
        # SHOW PREDICT.PY OUTPUT
        # ----------------------------------

        print()
        print("========== TFT OUTPUT ==========")
        print(result.stdout)
        print("=================================")


        if result.stderr:

            print()
            print("========== TFT ERROR ==========")
            print(result.stderr)
            print("===============================")


        # ----------------------------------
        # FIND PREDICTION VALUE
        # ----------------------------------

        prediction = None

        patterns = [

            r"Predicted\s+traffic\s*:\s*([-+]?\d+(?:\.\d+)?)",

            r"Predicted\s+traffic\s+volume\s*:\s*([-+]?\d+(?:\.\d+)?)",

            r"TRAFFIC\s+VOLUME\s+PREDICTION.*?"
            r"Predicted\s+traffic\s*:\s*"
            r"([-+]?\d+(?:\.\d+)?)"

        ]


        for pattern in patterns:

            match = re.search(
                pattern,
                result.stdout,
                re.IGNORECASE |
                re.DOTALL
            )

            if match:

                prediction = float(
                    match.group(1)
                )

                break


        # ----------------------------------
        # CHECK PREDICTION
        # ----------------------------------

        if prediction is None:

            return render_template(
                "index.html",

                current_date=datetime.now().strftime(
                    "%d %B %Y"
                ),

                current_time=datetime.now().strftime(
                    "%I:%M:%S %p"
                ),

                location=location,

                latitude=latitude,

                longitude=longitude,

                holiday=holiday,

                error="Prediction value could not be read."
            )


        # ----------------------------------
        # TRAFFIC CLASSIFICATION
        # ----------------------------------

        if prediction < 1200:

            traffic_level = "LOW TRAFFIC"

        elif prediction <= 4930:

            traffic_level = "MEDIUM TRAFFIC"

        else:

            traffic_level = "HEAVY TRAFFIC"


        # ----------------------------------
        # RETURN RESULT
        # ----------------------------------

        return render_template(

            "index.html",

            current_date=datetime.now().strftime(
                "%d %B %Y"
            ),

            current_time=datetime.now().strftime(
                "%I:%M:%S %p"
            ),

            prediction=round(
                prediction,
                2
            ),

            traffic_level=traffic_level,

            location=location,

            latitude=latitude,

            longitude=longitude,

            holiday=holiday
        )


    except Exception as e:

        print()
        print("Prediction error:")
        print(e)

        return render_template(

            "index.html",

            current_date=datetime.now().strftime(
                "%d %B %Y"
            ),

            current_time=datetime.now().strftime(
                "%I:%M:%S %p"
            ),

            location=location,

            latitude=latitude,

            longitude=longitude,

            holiday=holiday,

            error=str(e)
        )


# ==========================================
# START FLASK
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False
    )