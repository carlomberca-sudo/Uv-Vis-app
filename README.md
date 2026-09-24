# Uv-Vis spectruscopy Streamlit App

A private Streamlit app for PLQY analysis.

## Files
- `app.py` - main app
- `requirements.txt` - Python dependencies
- `.gitignore` - ignores local env and secrets

## Local run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Application for processing UV-Vis spectroscopy data, including spectral visualisation, comparison between samples, and extraction of relevant absorption/transmission metrics.
