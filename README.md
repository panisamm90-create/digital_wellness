# Digital Wellness Analyzer — Assembled Project

این نسخه UI موجود را به مدل‌های ML و سیستم Recommendation داخل نوت‌بوک متصل می‌کند.
 

- `templates/index.html` 
- `app.py` 
- `EDA.ipynb` 
- `dataset/digital_wellness_dataset.csv`  
- `requirements.txt`  

## اجرا
```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
 `http://127.0.0.1:5000` 

## اتصال ML
نوت‌بوک دو مدل دارد: `RandomForestClassifier` برای `unhealthy_day` و مدل XGBoost Regressor برای `total_screen_minutes`. API در صورت وجود دیتاست، همین دو جریان را در backend اجرا می‌کند. اگر دیتاست همراه پروژه نباشد، برنامه با fallback داده‌های پایه UI را بالا می‌آورد تا پروژه بدون crash اجرا شود.

## نکته
فایل دیتاست در نوت‌بوک با مسیر `dataset/digital_wellness_dataset.csv` ارجاع شده بود؛ چون این فایل در دو فایل ارسالی موجود نبود، آن را حدس نزدم یا از منبع دیگری جایگزین نکردم.
