# TrafficVision — 24-hour TFT forecast

## Folder structure

```text
TrafficVision_24h/
├── app.py
├── predict.py
├── train_tft.py
├── requirements.txt
├── dataset/
│   └── Metro_Interstate_Traffic_Volume_fixed.csv
├── models/
└── templates/
    └── index.html
└── static/
    ├── app.js
    └── style.css
```

## Setup on Windows

1. Copy `Metro_Interstate_Traffic_Volume_fixed.csv` into `dataset/`.
2. Open a terminal in this folder.
3. Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch Forecasting dependencies can vary by Python and PyTorch versions. If installation fails, use Python 3.10 or 3.11 in a fresh virtual environment and follow the current PyTorch installation instructions for your machine.

4. Train the **24-hour** model (this may take time):

```powershell
python train_tft.py
```

5. Start the website:

```powershell
python app.py
```

6. Open `http://127.0.0.1:5000`.

## Notes and limitations

- The model must be retrained using `train_tft.py`. The new checkpoint is `models/tft_traffic_volume_24h.ckpt`; a one-hour checkpoint is not interchangeable.
- Browser GPS requires user permission. `localhost` / `127.0.0.1` are considered secure contexts by modern browsers.
- Location search uses OpenStreetMap Nominatim (with Open-Meteo as fallback) and weather uses Open-Meteo; both require internet access. If a search fails, the Flask terminal prints a `[geocode]` line saying which service failed and why.
- The included Metro Interstate Traffic Volume dataset represents a US I-94 road sensor and ends in 2018. Live GPS and weather do **not** turn that into measured Hyderabad traffic. The best-time result is a model estimate learned from the I-94 dataset until Hyderabad traffic counts are added and the model is retrained.
- The festival flag is passed as a holiday-like model feature. The model has no separate learned festival effect unless the training dataset contains relevant festival labels. It should not be presented as a validated festival impact.
- Weather forecasts are used when available, but this model was trained on a historical US dataset; weather units are converted from Celsius to Kelvin for the Metro dataset's `temp` feature.
