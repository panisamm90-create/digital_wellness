from pathlib import Path
from typing import Any
import math
import pandas as pd
from flask import Flask, jsonify, request, render_template
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False

BASE = Path(__file__).resolve().parent
DATA = BASE / 'dataset' / 'digital_wellness_dataset.csv'
app = Flask(__name__, template_folder='templates', static_folder='static')

MODEL_STATE: dict[str, Any] = {'ready': False, 'source': 'fallback'}


def load_dataset():
    if not DATA.exists():
        return None
    df = pd.read_csv(DATA)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.day
    if 'age_group' in df.columns:
        age_order = ['13-15', '16-18', '19-24', '25-34', '35-50']
        enc = OrdinalEncoder(categories=[age_order], handle_unknown='use_encoded_value', unknown_value=-1)
        df['age_group_encoded'] = enc.fit_transform(df[['age_group']])
    return df


def fit_models(df):
    # Same two models as the notebook.
    clf_X = df.drop(columns=['unhealthy_day'])
    clf_y = df['unhealthy_day']
    reg_X = df.drop(columns=['total_screen_minutes'])
    reg_y = df['total_screen_minutes']

    def make_pipeline(X, model):
        cat = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        num = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        prep = ColumnTransformer([
            ('num', StandardScaler(), num),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat)
        ], remainder='passthrough')
        return Pipeline([('preprocess', prep), ('model', model)])

    clf = make_pipeline(clf_X, RandomForestClassifier(n_estimators=200, max_depth=9, random_state=42))
    clf.fit(clf_X, clf_y)

    if XGB_OK:
        reg_model = XGBRegressor(n_estimators=200, max_depth=9, random_state=42, objective='reg:squarederror')
    else:
        from sklearn.ensemble import RandomForestRegressor
        reg_model = RandomForestRegressor(n_estimators=200, max_depth=9, random_state=42)
    reg = make_pipeline(reg_X, reg_model)
    reg.fit(reg_X, reg_y)
    return clf, reg


def fallback_analysis(df=None):
    if df is not None and len(df):
        row = df.sort_values(['user_id', 'date']).iloc[-1].copy()
        total = float(row.get('total_screen_minutes', 273.6))
        social = float(row.get('social_minutes', 87.0))
        sleep = float(row.get('sleep_hours', 6.46))
        late = float(row.get('late_night_minutes', 16.2))
        unhealthy_prob = min(0.95, max(0.05, 0.35 + max(total-240,0)/600 + max(6-sleep,0)*0.08))
        pred = max(0, total * (1 + (social-90)/1200))
        label = 'unhealthy' if unhealthy_prob >= .5 else 'healthy'
        return row, label, unhealthy_prob if label == 'unhealthy' else 1-unhealthy_prob, pred
    row = pd.Series({
        'user_id': 1, 'age_group': '15-18', 'primary_device': 'smartphone',
        'social_minutes': 87.0, 'entertainment_minutes': 66.5, 'gaming_minutes': 35.6,
        'productivity_minutes': 60.7, 'education_minutes': 23.8,
        'total_screen_minutes': 273.6, 'notification_count': 77, 'pickup_count': 63,
        'late_night_minutes': 16.2, 'sleep_hours': 6.46,
        'longest_session_minutes': 30.7, 'avg_session_minutes': 13.2,
        'prev_day_total_minutes': 272.0, 'rolling3_avg_total_minutes': 271.9,
        'rolling7_avg_sleep_hours': 6.46,
    })
    return row, 'unhealthy', .72, 282.0


def train_if_possible():
    df = load_dataset()
    if df is None or len(df) < 20:
        MODEL_STATE.update({'ready': False, 'source': 'fallback', 'df': df})
        return
    try:
        clf, reg = fit_models(df)
        MODEL_STATE.update({'ready': True, 'source': 'notebook_ml', 'df': df, 'clf': clf, 'reg': reg})
    except Exception as exc:
        MODEL_STATE.update({'ready': False, 'source': 'fallback', 'df': df, 'error': str(exc)})


def make_analysis(payload):
    df = MODEL_STATE.get('df')
    if MODEL_STATE.get('ready'):
        row = df.sort_values(['user_id', 'date']).iloc[-1].copy()
        Xc = row.drop(labels=['unhealthy_day']).to_frame().T
        Xr = row.drop(labels=['total_screen_minutes']).to_frame().T
        try:
            proba = MODEL_STATE['clf'].predict_proba(Xc)[0]
            classes = MODEL_STATE['clf'].named_steps['model'].classes_
            p_unhealthy = float(proba[list(classes).index(1)]) if 1 in classes else 0.5
            label = 'unhealthy' if p_unhealthy >= .5 else 'healthy'
            conf = p_unhealthy if label == 'unhealthy' else 1-p_unhealthy
            pred = float(MODEL_STATE['reg'].predict(Xr)[0])
        except Exception:
            row, label, conf, pred = fallback_analysis(df)
    else:
        row, label, conf, pred = fallback_analysis(df)

    # Use the sleep value entered in the UI when it is available.
    if payload.get('sleep_hours') is not None:
        try:
            row['sleep_hours'] = float(payload['sleep_hours'])
        except (TypeError, ValueError):
            pass

    total = float(row.get('total_screen_minutes', 273.6))
    social = float(row.get('social_minutes', 87.0))
    entertainment = float(row.get('entertainment_minutes', 66.5))
    gaming = float(row.get('gaming_minutes', 35.6))
    productivity = float(row.get('productivity_minutes', 60.7))
    sleep = float(row.get('sleep_hours', 6.46))
    late = float(row.get('late_night_minutes', 16.2))

    # Keep the score simple and easy to explain in the UI.
    score = round(max(0, min(100, 55 + productivity/8 + sleep*3 - (social+entertainment+gaming)/18 - late/6)))
    suggestions = []
    if label == 'unhealthy':
        suggestions.append({'icon':'⚠️','text':'مدل امروز را خارج از محدوده‌ی سالم تشخیص داده است؛ زمان استفاده‌ی غیرضروری را کمی کاهش بده.'})
    else:
        suggestions.append({'icon':'🌿','text':'الگوی امروز در محدوده‌ی سالم قرار دارد؛ همین روند را حفظ کن.'})
    if social > 120:
        suggestions.append({'icon':'📵','text':'زمان شبکه‌های اجتماعی بالاست؛ برای این دسته از اپ‌ها محدودیت زمانی بگذار.'})
    if sleep < 6:
        suggestions.append({'icon':'🌙','text':'خواب ثبت‌شده کم است؛ زمان خواب منظم‌تری برای شب در نظر بگیر.'})
    if late > 60:
        suggestions.append({'icon':'🌙','text':'استفاده‌ی شبانه زیاد شده؛ نزدیک زمان خواب اعلان‌ها را کمتر کن.'})
    if pred > total * 1.1:
        suggestions.append({'icon':'📈','text':'مدل افزایش استفاده‌ی فردا را پیش‌بینی می‌کند؛ از امروز یک هدف مشخص برای فردا تعیین کن.'})
    if len(suggestions) < 3:
        suggestions.append({'icon':'💡','text':'هر ۵۰ دقیقه کار، چند دقیقه از صفحه فاصله بگیر و استراحت آگاهانه داشته باش.'})

    apps = [
        {'name':'شبکه‌های اجتماعی','icon':'📸','hours':round(social/60,1),'color':'pink'},
        {'name':'سرگرمی','icon':'▶️','hours':round(entertainment/60,1),'color':'purple'},
        {'name':'بازی','icon':'🎮','hours':round(gaming/60,1),'color':'amber'},
        {'name':'کار و آموزش','icon':'💻','hours':round((productivity+float(row.get('education_minutes',23.8)))/60,1),'color':'teal'},
    ]
    goal_target = float(payload.get('goal_target') or 2)
    current_hours = round(productivity/60, 1)
    used_apps = [a['name'] for a in sorted(apps, key=lambda x:x['hours'], reverse=True)[:3]]
    rolling_total = float(row.get('rolling3_avg_total_minutes', total))
    rolling_social = float(row.get('social_minutes', social))
    personalized = []
    if rolling_total and total/rolling_total > 1.3:
        personalized.append(f'امروز زمان صفحه‌نمایش {total/rolling_total:.1f} برابر میانگین معمول ثبت شده است؛ فردا آن را به سطح معمول نزدیک‌تر کن.')
    if rolling_social and social/rolling_social > 1.3:
        personalized.append(f'مصرف شبکه‌های اجتماعی نسبت به میانگین اخیر افزایش داشته است (اپ‌های شاخص: {", ".join(used_apps)}).')

    return {
        'balance_score': score,
        'usage': {'work_hours': round(productivity/60,1), 'fun_hours': round((entertainment+gaming)/60,1), 'rest_hours': round(sleep,1)},
        'alert': {'minutes': round(late)},
        'suggestions': suggestions[:5],
        'apps': apps,
        'goal': {'target_hours': goal_target, 'current_hours': current_hours},
        'prediction': {'text': str(round(pred/60,1)), 'tomorrow_screen_minutes': round(pred,1)},
        'ml': {'label': label, 'confidence': round(conf,3), 'backend': MODEL_STATE.get('source')},
        'personalization': personalized,
    }


@app.get('/')
def index():
    return render_template('index.html')

@app.get('/api/health')
def health():
    return jsonify({'ok': True, 'model': MODEL_STATE.get('source'), 'xgboost': XGB_OK})

@app.post('/api/analyze')
def analyze():
    payload = request.get_json(silent=True) or {}
    return jsonify(make_analysis(payload))

@app.post('/api/retrain')
def retrain():
    train_if_possible()
    return jsonify({'ok': True, 'model': MODEL_STATE.get('source')})

train_if_possible()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
