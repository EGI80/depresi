import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE

# Set page config at the very top
st.set_page_config(
    page_title="Prediksi Depresi Mahasiswa",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load data with caching for performance
@st.cache_data(show_spinner=True)
def load_data():
    data = pd.read_csv('depresi/Student_Mental_Health.csv')

    imputer = SimpleImputer(strategy='most_frequent')
    data[['Age', 'What is your course?', 'Marital status']] = imputer.fit_transform(
        data[['Age', 'What is your course?', 'Marital status']]
    )

    data.dropna(subset=[
        'Do you have Depression?', 'Choose your gender', 'Do you have Anxiety?',
        'Do you have Panic attack?', 'Did you seek any specialist for a treatment?'
    ], inplace=True)

    data['Your current year of Study'] = data['Your current year of Study'].str.lower().str.strip()
    data['What is your CGPA?'] = data['What is your CGPA?'].str.replace(' ', '')
    data['What is your CGPA?'] = data['What is your CGPA?'].str.replace('-', ' to ')

    data['CGPA_numeric'] = data['What is your CGPA?'].apply(
        lambda x: np.mean(list(map(float, x.split(' to ')))) if 'to' in x else float(x)
    )

    Q1 = data['CGPA_numeric'].quantile(0.25)
    Q3 = data['CGPA_numeric'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_condition = (data['CGPA_numeric'] < (Q1 - 1.5 * IQR)) | (data['CGPA_numeric'] > (Q3 + 1.5 * IQR))
    data = data[~outlier_condition]

    mapping_binary = {'Yes': 1, 'No': 0}
    data['Do you have Depression?'] = data['Do you have Depression?'].map(mapping_binary).astype(int)
    data['Choose your gender'] = data['Choose your gender'].map({'Male': 1, 'Female': 0}).astype(int)

    for col in ['Marital status', 'Do you have Anxiety?', 'Do you have Panic attack?', 'Did you seek any specialist for a treatment?']:
        data[col] = data[col].map(mapping_binary).astype(int)

    data['Your current year of Study'] = data['Your current year of Study'].map({
        'year 1': 1,
        'year 2': 2,
        'year 3': 3,
        'year 4': 4
    })

    data.drop(columns=['Timestamp', 'What is your course?', 'What is your CGPA?'], inplace=True)
    return data

data = load_data()

X = data[['Choose your gender', 'Age', 'Your current year of Study', 'CGPA_numeric', 'Marital status',
          'Do you have Anxiety?', 'Do you have Panic attack?', 'Did you seek any specialist for a treatment?']]
y = data['Do you have Depression?']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

rf_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf_estimator, n_features_to_select=5)
rfe.fit(X_train, y_train)

X_train_selected = rfe.transform(X_train)
X_test_selected = rfe.transform(X_test)

param_grid = {
    'n_estimators': [100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    verbose=0,
    n_jobs=-1
)
rf_grid.fit(X_train_selected, y_train)

best_rf_model = RandomForestClassifier(**rf_grid.best_params_, random_state=42)
best_rf_model.fit(X_train_selected, y_train)

# ----- UI Design with Friendly and Elegant Style -----

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        /* Global styles */
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
            background: #f5f7fa;
            color: #1f2937;
        }

        /* Center container */
        .main-container {
            max-width: 480px;
            margin: 40px auto 60px;
            background: white;
            padding: 32px 36px;
            border-radius: 20px;
            box-shadow: 0 8px 20px rgba(100, 116, 139, 0.15);
        }

        /* Header */
        header.app-header {
            position: sticky;
            top: 0;
            background: rgba(255 255 255 / 0.8);
            backdrop-filter: saturate(180%) blur(12px);
            box-shadow: 0 1px 10px rgba(0, 0, 0, 0.05);
            padding: 16px 0;
            text-align: center;
            font-size: 1.75rem;
            font-weight: 700;
            color: #4F46E5;
            border-radius: 0 0 24px 24px;
            user-select: none;
        }

        /* Form labels */
        label {
            font-weight: 600;
            margin-bottom: 6px;
            display: block;
            color: #374151;
        }

        /* Inputs and selects */
        .stNumberInput>div>div>input,
        .stSelectbox>div>div>div>div {
            padding: 12px 14px !important;
            border: 1.5px solid #d1d5db !important;
            border-radius: 12px !important;
            transition: border-color 0.3s ease;
        }

        .stNumberInput>div>div>input:focus,
        .stSelectbox>div>div>div>div:focus {
            border-color: #6366F1 !important;
            outline: none !important;
            box-shadow: 0 0 8px rgb(99 102 241 / 0.4);
        }

        /* Button styles */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #6366F1, #4F46E5);
            color: white;
            font-weight: 700;
            border-radius: 16px;
            padding: 16px 0;
            font-size: 1.1rem;
            transition: box-shadow 0.3s ease, transform 0.2s ease;
            width: 100%;
            border: none;
            cursor: pointer;
            user-select: none;
        }

        div.stButton > button:first-child:hover {
            box-shadow: 0 0 15px rgb(79 70 229 / 0.6);
            transform: scale(1.03);
        }
        div.stButton > button:first-child:focus {
            outline: none !important;
            box-shadow: 0 0 15px rgb(79 70 229 / 0.9);
        }

        /* Result message */
        .result-message {
            margin-top: 24px;
            padding: 20px 24px;
            border-radius: 16px;
            font-size: 1.15rem;
            text-align: center;
            font-weight: 600;
            user-select: none;
        }

        .result-success {
            background: #d1fae5;
            color: #065f46;
        }

        .result-fail {
            background: #fee2e2;
            color: #991b1b;
        }

        /* Responsive adjustments */
        @media (max-width: 480px) {
            .main-container {
                margin: 20px 16px 40px;
                padding: 28px 24px;
            }
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<header class="app-header">Prediksi Depresi Mahasiswa</header>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    gender = st.selectbox("Pilih Jenis Kelamin", ["Male", "Female"])
    age = st.number_input("Masukkan Usia", min_value=18, max_value=100, value=20)
    year_of_study = st.selectbox("Tahun Studi Saat Ini", ["year 1", "year 2", "year 3", "year 4"])
    year_of_study_numeric = 1 if year_of_study == "year 1" else 2 if year_of_study == "year 2" else 3 if year_of_study == "year 3" else 4
    cgpa = st.number_input("Masukkan CGPA", min_value=0.0, max_value=4.0, value=3.0, format="%.2f")
    marital_status = st.selectbox("Status Perkawinan", ["Single", "Married"])
    anxiety = st.selectbox("Apakah Anda Mengalami Kecemasan?", ["Yes", "No"])
    panic_attack = st.selectbox("Apakah Anda Mengalami Serangan Panik?", ["Yes", "No"])
    specialist_treatment = st.selectbox("Apakah Anda Mencari Pengobatan?", ["Yes", "No"])

    if st.button("Prediksi"):
        input_data = np.array([[1 if gender == "Male" else 0,
                                age,
                                year_of_study_numeric,
                                cgpa,
                                1 if marital_status == "Married" else 0,
                                1 if anxiety == "Yes" else 0,
                                1 if panic_attack == "Yes" else 0,
                                1 if specialist_treatment == "Yes" else 0]])
        input_data_selected = rfe.transform(input_data)

        prediction = best_rf_model.predict(input_data_selected)

        if prediction[0] == 1:
            st.markdown('<div class="result-message result-fail">Prediksi: Anda mungkin mengalami depresi.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-message result-success">Prediksi: Anda tidak mengalami depresi.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

