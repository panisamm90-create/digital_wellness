
Digital Wellness Analyzer — Assembled Project
This version connects the existing UI to the machine learning models and recommendation system in the notebook.
Project Files
templates/index.html
app.py
EDA.ipynb
dataset/digital_wellness_dataset.csv
requirements.txt
How to Run
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
python app.py
Open http://127.0.0.1:5000 in your browser.
Machine Learning Integration
The notebook contains two models: a RandomForestClassifier for unhealthy_day and an XGBoost Regressor for total_screen_minutes. When the dataset is available, the API runs both workflows in the backend. If the dataset is missing, the application uses fallback data to launch the basic UI without crashing.
Note
The notebook references the dataset at dataset/digital_wellness_dataset.csv. Since this file was not included in the two supplied files, it was not guessed or replaced with data from another source.
# project files
you can access the dataset files through Google Drive:
[download dataset file](https://drive.google.com/file/d/1tXchywP_eRNdNCy-EG0h8Sp9-_WZEla9/view?usp=share_link)
