# Analytics portfolio: Streamlit app

Six projects in one app: Turtle Games (LSE 3), ConnectTel churn (LSE 4), NHS capacity (LSE 2),
2Market (LSE 1), banking transaction anomalies, and option-price verification.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo (app.py, requirements.txt, data/, .streamlit/).
2. On share.streamlit.io choose "New app", pick the repo, set the main file to app.py.

`prep_data.py` is the one-off script that shrank the raw NHS, marketing and banking files into the small CSVs in `data/`.
It is not needed to run the app. Large raw files and the project videos are not part of the app.
