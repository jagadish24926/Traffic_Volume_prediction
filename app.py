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

    return render_template(
        "index.html",
        current_date=datetime.now().strftime("%d %B %Y"),
        current_time=datetime.now().strftime("%I:%M:%S %p")
    )


# ==========================================
# PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    location = request.form.get(
        "location",
        "I-94 — Minneapolis"
    )

    holiday = request.form.get(
        "holiday",
        "No Holiday"
    )

    try:

        # ==========================================
        # RUN TFT PREDICTION
        # ==========================================

        result = subprocess.run(
            [sys.executable, "predict.py"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        output = result.stdout
        error = result.stderr


        # ==========================================
        # CHECK FOR PYTHON ERROR
        # ==========================================

        if result.returncode != 0:

            return render_template(
                "index.html",

                current_date=datetime.now().strftime(
                    "%d %B %Y"
                ),

                current_time=datetime.now().strftime(
                    "%I:%M:%S %p"
                ),

                location=location,

                holiday=holiday,

                error="TFT prediction failed.",

                details=error[-4000:]
            )


        # ==========================================
        # EXTRACT PREDICTION
        # ==========================================

        prediction = None


        # Your predict.py currently prints:
        #
        # Predicted traffic volume : 623.17 vehicles/hour
        #
        # So we search for the number after
        # "Predicted traffic volume"

        pattern = (
            r"Predicted\s+traffic\s+volume"
            r"\s*:\s*"
            r"([-+]?\d+(?:\.\d+)?)"
        )


        match = re.search(
            pattern,
            output,
            re.IGNORECASE
        )


        if match:

            prediction = float(
                match.group(1)
            )


        # ==========================================
        # FALLBACK
        # ==========================================

        # If the first pattern doesn't find it,
        # search for any prediction-like line.

        if prediction is None:

            fallback_patterns = [

                r"prediction\s*:\s*"
                r"([-+]?\d+(?:\.\d+)?)",

                r"predicted\s*:\s*"
                r"([-+]?\d+(?:\.\d+)?)",

                r"traffic\s+volume\s*:\s*"
                r"([-+]?\d+(?:\.\d+)?)"

            ]


            for fallback_pattern in fallback_patterns:

                match = re.search(
                    fallback_pattern,
                    output,
                    re.IGNORECASE
                )

                if match:

                    prediction = float(
                        match.group(1)
                    )

                    break


        # ==========================================
        # STILL NOT FOUND
        # ==========================================

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

                holiday=holiday,

                error="Prediction value could not be read.",

                details=output[-5000:]

            )


        # ==========================================
        # TRAFFIC CLASSIFICATION
        # ==========================================

        if prediction < 1200:

            traffic_status = "LOW"

            traffic_class = "low"

            traffic_icon = "🟢"


        elif prediction <= 4930:

            traffic_status = "MEDIUM"

            traffic_class = "medium"

            traffic_icon = "🟠"


        else:

            traffic_status = "HEAVY"

            traffic_class = "heavy"

            traffic_icon = "🔴"


        # ==========================================
        # FORECAST TIME
        # ==========================================

        forecast_time = datetime.now().strftime(
            "%I:%M %p"
        )


        # ==========================================
        # SHOW RESULT
        # ==========================================

        return render_template(

            "index.html",

            current_date=datetime.now().strftime(
                "%d %B %Y"
            ),

            current_time=datetime.now().strftime(
                "%I:%M:%S %p"
            ),

            location=location,

            holiday=holiday,

            prediction=f"{prediction:,.2f}",

            traffic_status=traffic_status,

            traffic_class=traffic_class,

            traffic_icon=traffic_icon,

            forecast_time=forecast_time

        )


    # ==========================================
    # GENERAL ERROR
    # ==========================================

    except Exception as e:

        return render_template(

            "index.html",

            current_date=datetime.now().strftime(
                "%d %B %Y"
            ),

            current_time=datetime.now().strftime(
                "%I:%M:%S %p"
            ),

            location=location,

            holiday=holiday,

            error=str(e)

        )


# ==========================================
# RUN FLASK
# ==========================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print(" SMART TRAFFIC PREDICTION SYSTEM")
    print("==========================================")
    print("Open: http://127.0.0.1:5000")
    print("==========================================")
    print()

    app.run(
        debug=True
    )