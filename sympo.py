# streamlit_symposium_code_editor.py
# -------------------------------------------------------------
# Streamlit-based coding challenge app for symposium model-building event
# Features
# - Login/registration page: collects team details (team name, lead email, phone, college)
#   Left panel shows event logo; right panel has form.
#   Details are saved on the HOST machine (where Streamlit app runs) in event_data/teams.csv
#   The team name is shown to the user after login throughout the app.
# - Problem menu: shows 6 problem statements. Teams are issued a problem code (PS1..PS6).
#   They must enter the correct code to open the corresponding problem.
# - Each problem shows: a question + starter (half-built) code in an editor (text area).
#   Participants must fill only the TODO parts. New imports / dangerous code are blocked.
# - Run button: executes the user code in a restricted sandbox and runs hidden tests.
#   Shows "✅ All tests passed" or descriptive errors.
# - Submit button: enabled only after tests pass; allowed ONCE per team per problem.
#   Saves the submitted code + team name + timestamp on HOST machine under event_data/submissions.
# - Admin page (optional): view registrations and submissions (requires admin password).
# -------------------------------------------------------------

import os
import io
import ast
import time
import textwrap
from datetime import datetime
from typing import Dict, Any, Tuple, List

import traceback
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import streamlit as st
import sklearn
from sklearn import datasets, model_selection, linear_model, metrics, preprocessing, pipeline,ensemble,naive_bayes,cluster,tree
from sklearn.ensemble import RandomForestClassifier


# -------------- Organizer configuration --------------
APP_TITLE = "DATAVERSE — Coding Challenge"
LOGO_PATH = "assets/logo.png"  # put your logo file here; or change path
DATA_DIR = "event_data"  # all saved files live here on the HOST machine
ADMIN_PASSWORD = os.environ.get("EVENT_ADMIN_PASSWORD", "admin123")  # change or set env var

# Problem definitions live here. 
# To customize: Edit each dict entry (title, prompt, starter_code, validator).
# - Keep function names consistent between starter_code and validator expectations.
# - You can replace the logic/tests inside validator() as per your dataset/model.

#Load the dataset
# Load heartrate dataset
DATA_PATH = "heartrate.csv"  # make sure file is in the same folder or give full path
try:
    data = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    data = pd.DataFrame()  # fallback


# ---- Problem PS1: Train/Test Split ----
PS1_STARTER = '''
"""
Predict whether a patient has heart disease from the heartrate dataset.
Dataset columns: age, bmi, systolic_bp, cholesterol_total, heart_disease

Target: heart_disease (0 = No, 1 = Yes)

Fill in the blanks to train a RandomForestClassifier model 
and evaluate accuracy and classification report.
"""


# Features and target
X = data[["age", "bmi", "systolic_bp", "cholesterol_total"]]
y = data["heart_disease"]

# TODO: split into train/test
X_train, X_test, y_train, y_test = ________________ ____________(X, y, test_size=0.2, random_state=42)

# TODO: create and train model
model = ____________________(random_state=42)
model.fit(__________, __________)

# TODO: make predictions
y_pred = model.predict(__________)

# Evaluate
print("Accuracy:",______________accuracy_score(y_test, y_pred))
print(_____________classification_report(y_test, y_pred))
'''

def PS1_validator(ns):
    required_vars = ["X_train", "X_test", "y_train", "y_test", "model", "y_pred"]
    for var in required_vars:
        if var not in ns or ns[var] is None:
            return False, f"{var} missing or None"

    # ✅ Check model type directly (RandomForestClassifier is in safe_globals already)
    if not isinstance(ns["model"], RandomForestClassifier):
        return False, "Model must be RandomForestClassifier"

    if len(ns["X_train"]) == 0 or len(ns["X_test"]) == 0:
        return False, "Train/test sets are empty"

    if len(ns["y_pred"]) != len(ns["y_test"]):
        return False, "Predictions length mismatch"

    # ✅ Use ns["metrics"] instead of raw import
    acc = metrics.accuracy_score(ns["y_test"], ns["y_pred"])
    if acc < 0.6:  # adjust threshold if dataset is harder
        return False, f"Accuracy too low: {acc:.2f}"
    
    return True, "Heart Disease Prediction task completed successfully!"


# ---- Problem PS2: Linear Regression ----
PS2_STARTER = '''
"""
Predict patient stroke risk (Low / Medium / High).
Features: age, blood_glucose, hypertension, diabetes
Target: stroke_risk (categorical)

Fill in the blanks to train a KNeighborsClassifier model 
for multi-class classification.
"""

X = data[['__________','_______','__________','___________']]
y = data['_______']

X_train, X_test, y_train, y_test = __________________train_test_split(X, y, test_size=________, random_state=____ )

model = ____________(n_neighbors=_____)
model.fit(____________, ________)
y_pred = model.predict(__________)

print("Accuracy:", ____________accuracy_score(y_test, y_pred))
print(___________classification_report(y_test, y_pred))

'''


def PS2_validator(ns):
    if "model" not in ns: 
        return False, "Model not defined"

    if ns["y"].name != "stroke_risk":
        return False, "Target column should be 'stroke_risk'"

    expected_cols = {"age","blood_glucose","hypertension","diabetes"}
    if set(ns["X"].columns) != expected_cols:
        return False, "Incorrect features selected"

    if ns["model"].__class__.__name__ != "KNeighborsClassifier":
        return False, "Model should be KNeighborsClassifier"

    if ns["model"].n_neighbors != 5:
        return False, "n_neighbors should be 5"

    return True, "Stroke risk prediction successful"


# ---- Problem PS3: Naive Bayes ----
PS3_STARTER = '''
"""
Predict chest pain type using Naive Bayes.
Columns->>'age','systolic_bp','cholesterol_total','chest_pain_type'
"""

X = data[['__________','__________','_________']]
y = data['____________']

X_train, X_test, y_train, y_test = ___________________train_test_split(X, y, test_size=_____________, random_state=______ )

model = ____________GaussianNB()
model.fit(_____________, ________)
y_pred = model.predict(____________)

print("Accuracy:", _____________accuracy_score(y_test, y_pred))
print(_____________classification_report(y_test, y_pred))
'''

def PS3_validator(ns):
    if "model" not in ns: 
        return False, "Model not defined"
    if ns["y"].name != "chest_pain_type":
        return False, "Target column should be 'chest_pain_type'"
    if set(ns["X"].columns) != {"age","systolic_bp","cholesterol_total"}:
        return False, "Incorrect features selected"
    if type(ns["model"]).__name__ != "GaussianNB":
        return False, "Model should be GaussianNB"
    return True, "Chest Pain Type prediction successful"



# ---- Problem PS5: Logistic Regression ----
PS4_STARTER = '''
"""
Predict arrhythmia risk (0 = No, 1 = Yes).
Features: heart_rate, age, ecg_results, bmi
Target: arrhythmia_risk (binary classification)

Fill in the blanks to train a LogisticRegression model 
with appropriate solver for binary classification.
"""

# Features and target
X = data[['___________','_________','___________','______']].copy()
y = data['____________']

# Encode categorical column
le = ____________.LabelEncoder()
X['ecg_results'] = le.___________(X['ecg_results'])

# Train/test split
X_train, X_test, y_train, y_test = _____________.train_test_split(
    X, y, test_size=_________, random_state=________
)

# Logistic Regression model
model = ______________LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(___________, ________)

# Predictions
y_pred = model.predict(_________)

print("Accuracy:", ____________accuracy_score(y_test, y_pred))
print(_____________classification_report(y_test, y_pred))

'''

def PS4_validator(ns):
    if "model" not in ns:
        return False, "Model not defined"

    if ns["y"].name != "arrhythmia_risk":
        return False, "Target column should be 'arrhythmia_risk'"

    expected_cols = {"heart_rate", "age", "ecg_results", "bmi"}
    if set(ns["X"].columns) != expected_cols:
        return False, "Incorrect features selected"

    # Ensure ecg_results is encoded (should be numeric, not string)
    if ns["X"]['ecg_results'].dtype not in ['int64', 'float64']:
        return False, "Feature 'ecg_results' must be label-encoded to numeric values"

    if ns["model"].__class__.__name__ != "LogisticRegression":
        return False, "Model should be LogisticRegression"

    return True, "Arrhythmia risk prediction successful"

# ---- Problem PS6: Decision Tree Classifier ----
PS5_STARTER = '''
"""
Predict hospitalization risk (Low / Medium / High).
Features: smoking_status, alcohol_consumption, physical_activity_hours_week, sleep_hours_night
Target: hospitalization_risk (categorical classification)

Fill in the blanks to train a DecisionTreeClassifier model 
for multiclass classification.
"""

# Features and target
X = data[['___________','________________','____________','____________']].copy()
y = data['__________']


for col in ['smoking_status','alcohol_consumption']:
    le = _______________LabelEncoder()
    X[col] = le._______________(X[col])


X_train, X_test, y_train, y_test = ____________train_test_split(X, y, test_size=_____, random_state=_____)


model =DecisionTreeClassifier(random_state=__________)
model.fit(__________, _____________)

# Predictions
y_pred = model.predict(__________)

print("Accuracy:", __________accuracy_score(y_test, y_pred))
print(_________classification_report(y_test, y_pred))

'''


def PS5_validator(ns):
    if "model" not in ns:
        return False, "Model not defined"

    if ns["y"].name != "hospitalization_risk":
        return False, "Target column should be 'hospitalization_risk'"

    expected_cols = {"smoking_status", "alcohol_consumption", "physical_activity_hours_week", "sleep_hours_night"}
    if set(ns["X"].columns) != expected_cols:
        return False, "Incorrect features selected"

    # Ensure categorical encoding for smoking_status and alcohol_consumption
    for col in ["smoking_status", "alcohol_consumption"]:
        if ns["X"][col].dtype not in ['int64', 'float64']:
            return False, f"Feature '{col}' must be label-encoded to numeric values"

    if ns["model"].__class__.__name__ != "DecisionTreeClassifier":
        return False, "Model should be DecisionTreeClassifier"

    return True, "Hospitalization risk prediction successful"

# ---- Final Problem Set ----
PROBLEMS = {
    "PS1": {"title": "Predict whether a patient has heart disease", "prompt": "Random Forest Classification model", "starter": PS1_STARTER, "validator": PS1_validator},
    "PS2": {"title": "Predict patient stroke risk ", "prompt": "KNearestNeighbor model", "starter": PS2_STARTER, "validator": PS2_validator},
    "PS3": {"title": "Predict chest pain type", "prompt": "Naive Bayes Model", "starter": PS3_STARTER, "validator": PS3_validator},
    "PS4": {"title": "Predict arrhythmia risk", "prompt": "Logistic Regression Model ", "starter": PS4_STARTER, "validator": PS4_validator},
    "PS5": {"title": "Predict hospitalization risk", "prompt": "Decision Tree Classifier Model", "starter": PS5_STARTER, "validator": PS5_validator},
}


ALLOWED_IMPORTS ={
    # Modules that participants are allowed to import IN ADDITION to those already present in starter code
    # (We will still block any new imports beyond these, to reduce access to prebuilt code.)
    'numpy', 'pandas', 're', 'collections',
    'sklearn', 'sklearn.datasets', 'sklearn.model_selection','sklearn.cluster','sklearn.tree', 'sklearn.linear_model', 'sklearn.metrics', 'sklearn.preprocessing','sklearn.naive_bayes', 'sklearn.pipeline','sklearn.ensemble','sklearn.ensemble.RandomForestClassifier'
}

# -------------- Utilities --------------

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'submissions'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'logs'), exist_ok=True)


def save_registration(team: Dict[str, str]):
    ensure_dirs()
    path = os.path.join(DATA_DIR, 'teams.csv')
    df_new = pd.DataFrame([team])
    if os.path.exists(path):
        df = pd.read_csv(path)
        # remove existing entry for same team_name
        df = df[df['team_name'] != team['team_name']]
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)


def load_submissions_df() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, 'submissions.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=['timestamp', 'team_name', 'problem_code', 'result', 'file_path'])


def already_submitted(team_name: str, problem_code: str) -> bool:
    df = load_submissions_df()
    row = df[(df['team_name'] == team_name) & (df['problem_code'] == problem_code)]
    return not row.empty


def log_submission(team_name: str, problem_code: str, success: bool, file_path: str):
    ensure_dirs()
    path = os.path.join(DATA_DIR, 'submissions.csv')
    df = load_submissions_df()
    df.loc[len(df)] = [datetime.now().isoformat(timespec='seconds'), team_name, problem_code, 'PASS' if success else 'FAIL', file_path]
    df.to_csv(path, index=False)


def sanitize_user_code(code: str) -> Tuple[bool, str]:
    """Reject code with disallowed imports or dangerous names. Return (ok, err_msg)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Build full module name
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name
                    if mod not in ALLOWED_IMPORTS:
                        return False, f"Import of '{mod}' not allowed"
            else:
                mod = node.module or ''
                if mod not in ALLOWED_IMPORTS:
                    return False, f"Import from '{mod}' not allowed"
        elif isinstance(node, ast.Call):
            # forbid os.system, subprocess, __import__, open, exec, eval etc.
            if isinstance(node.func, ast.Name) and node.func.id in {'exec', 'eval', '__import__', 'open'}:
                return False, f"Use of '{node.func.id}' not allowed"
            if isinstance(node.func, ast.Attribute):
                attr = f"{getattr(node.func.value, 'id', '')}.{node.func.attr}"
                if attr in {'os.system', 'os.popen', 'subprocess.Popen', 'subprocess.call', 'subprocess.run'}:
                    return False, f"Use of '{attr}' not allowed"
        elif isinstance(node, ast.Attribute):
            # block dunder access like object.__subclasses__
            if node.attr.startswith('__') and node.attr.endswith('__'):
                return False, "Dunder attribute access is not allowed"
    return True, "OK"

def run_user_code(code: str) :
    """Execute user code in restricted namespace. Return (ok, namespace, err)."""
    # Restricted globals/locals
    safe_globals = {"__builtins__" :{
            "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
            "range": range, "enumerate": enumerate,
            "float": float, "int": int, "str": str,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            "print": print, "zip": zip, "map": map, "filter": filter,
            "any": any, "all": all, "sorted": sorted,
        },

        # expose common packages
         "np": np,
        "pd": pd,
        "sklearn": __import__("sklearn"),
        "model_selection": model_selection,
        "linear_model": linear_model,
        "tree":tree,
        "naive_bayes":naive_bayes,
        "metrics": metrics,
        "ensemble":ensemble,
        "cluster":cluster,
        "preprocessing": preprocessing,
        "RandomForestClassifier":RandomForestClassifier,
        "pipeline": pipeline,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "KNeighborsClassifier": KNeighborsClassifier,

        #expose data
        "data":data
    }
    safe_locals={}
    try:
        exec(compile(code, '<user_code>', 'exec'), safe_globals, safe_locals)
        # merge dicts so validators can see everything
        ns = {**safe_globals, **safe_locals}
        return True, ns, ""
    except Exception as e:
        return False, {}, f"Execution error: {e}\n{traceback.format_exc()}"

# -------------- Streamlit UI --------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# Persistent session state
if 'team' not in st.session_state:
    st.session_state.team = None  # dict with details
if 'passed' not in st.session_state:
    st.session_state.passed = {}  # problem_code -> bool

# Sidebar navigation
pages = ["Login", "Problems"]
page = st.sidebar.radio("Navigate", pages)

# Display team name in sidebar if logged in
if st.session_state.team:
    st.sidebar.success(f"Team: {st.session_state.team.get('team_name')}")

# -------------- Login Page --------------
if page == "Login":
    left, right = st.columns([1, 1])
    
    with left:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)

        else:
            st.info(f"Place your logo at: {LOGO_PATH}")
    with right:
        st.header("Team Registration")
        with st.form("reg_form", clear_on_submit=False):
            team_name = st.text_input("Team Name*")
            lead_name = st.text_input("Team Lead Name*")
            email = st.text_input("Lead Email*")
            phone = st.text_input("Phone Number*")
            college = st.text_input("College Name*")
            submitted = st.form_submit_button("Save & Enter App")
        if submitted:
            if not (team_name and lead_name and email and phone and college):
                st.error("Please fill all required fields.")
            else:
                team = {
                    'team_name': team_name.strip(),
                    'lead_name': lead_name.strip(),
                    'email': email.strip(),
                    'phone': phone.strip(),
                    'college': college.strip(),
                    'registered_at': datetime.now().isoformat(timespec='seconds')
                }
                save_registration(team)
                st.session_state.team = team
                st.success(f"Welcome, {team['team_name']}! Your details are saved on the host machine.")
# ---------------- Problems Page ----------------
elif page == "Problems":
    if not st.session_state.team:
        st.warning("Please register/login first.")
        st.stop()

    st.subheader("Open Your Assigned Problem")
    st.write("Enter the problem code provided by the organizers (e.g., PS1..PS5).")

    colA, colB = st.columns([1,2])
    with colA:
        problem_code = st.text_input("Problem Code", value="PS1").strip().upper()
        open_btn = st.button("Open Problem")
    with colB:
        st.info("Once opened, you will see the question and starter code. Fill only the TODO parts. New imports or unsafe code are blocked.")

    if open_btn:
        st.session_state.current_problem = problem_code

    problem_code = st.session_state.get('current_problem')

    if problem_code and problem_code in PROBLEMS:
        meta = PROBLEMS[problem_code]
        st.markdown(f"### {problem_code}: {meta['title']}")

        # Tabs for Code and Description
        tab_code, tab_desc = st.tabs(["Code Editor", "Description"])

        # -------------------- CODE TAB --------------------
        with tab_code:
            with st.expander("Problem Description", expanded=True):
                st.write(meta['starter'].split('\n', 6)[0].strip('\n') or meta['prompt'])
                st.write(meta['prompt'])

            # Starter code editor
            if f"code_{problem_code}" not in st.session_state:
                st.session_state[f"code_{problem_code}"] = meta['starter']

            code_text = st.text_area(
                label="Your Code (edit the TODOs only)",
                value=st.session_state[f"code_{problem_code}"],
                height=420,
                key=f"editor_{problem_code}",
            )
            st.session_state[f"code_{problem_code}"] = code_text

            # Buttons
            run_col, submit_col, status_col = st.columns([1,1,2])
            with run_col:
                run_clicked = st.button("▶ Run Tests", key=f"run_{problem_code}")
            with submit_col:
                disable_submit = already_submitted(st.session_state.team['team_name'], problem_code)
                submit_clicked = st.button("✅ Submit (one-time)", key=f"submit_{problem_code}", disabled=disable_submit)
            with status_col:
                if disable_submit:
                    st.info("You have already submitted this problem. Submit is disabled.")

            # Process run
            if run_clicked:
                ok, msg = sanitize_user_code(code_text)
                if not ok:
                    st.error(msg)
                else:
                    ok2, ns, err = run_user_code(code_text)
                    if not ok2:
                        st.error(err)
                    else:
                        validator = meta['validator']
                        passed, detail = validator(ns)
                        st.session_state.passed[problem_code] = passed
                        if passed:
                            st.success("✅ All tests passed! You can now submit.")
                        else:
                            st.error("❌ Tests failed: " + detail)

            # Process submit
            if submit_clicked:
                if not st.session_state.passed.get(problem_code, False):
                    st.error("Please run and pass the tests before submitting.")
                else:
                    ensure_dirs()
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    fname = f"{st.session_state.team['team_name']}_{problem_code}_{ts}.py"
                    fpath = os.path.join(DATA_DIR,'submissions',fname)
                    with open(fpath,'w',encoding='utf-8') as f:
                        f.write(code_text)
                    log_submission(st.session_state.team['team_name'], problem_code, True, fpath)
                    st.success("🎉 Submission saved on host machine. Good luck!")
                    st.session_state.submitted_file_path = fpath
                    st.rerun()  # refresh to enable description tab

                # -------------------- DESCRIPTION TAB --------------------
        with tab_desc:
            if already_submitted(st.session_state.team['team_name'], problem_code):
                desc_key = f"desc_{problem_code}"
                if desc_key not in st.session_state:
                    st.session_state[desc_key] = ""  # initialize if not present

                # Display the question for this problem
                PROBLEMS = {
    "PS1": {
       "title": "Predict whether a patient has heart disease", "prompt": "Explain the process of Machine Learning concepts in detail",
        "starter": PS1_STARTER,
        "validator": PS1_validator
    },
    "PS2": {
       "title": "Predict patient stroke risk ", 
       "prompt": "What is dimensionality reduction? what are its method and why it is important",
        "starter": PS2_STARTER,
        "validator": PS2_validator
    },
    "PS3": {
        "title": "Predict chest pain type",
        "prompt": "Explain Bias Variance Trade off  what does  high variance and low bias leads to ?? ",
        "starter": PS3_STARTER,
        "validator": PS3_validator
    },

    "PS4": {
       "title": "Predict arrhythmia risk",
        "prompt": "explain Reinforcement learning",
        "starter": PS4_STARTER,
        "validator": PS4_validator
    },

    "PS5": {
        "title": "Predict hospitalization risk", 
        "prompt": "explain Over fitting and Under fitting ",
        "starter": PS5_STARTER,
        "validator": PS5_validator
    },

    
}


                st.markdown(f"### Description / Explanation for {problem_code}")
                st.write(PROBLEMS[problem_code]['prompt'])  # show the problem-specific question

                # Form for optional description
                with st.form(f"description_form_{problem_code}"):
                    desc_input = st.text_area(
                        label="Answer the above question",
                        value=st.session_state[desc_key],
                        height=200,
                        key=f"desc_input_{problem_code}"  # different key for widget to avoid conflict
                    )
                    desc_submitted = st.form_submit_button("Save Description")
                    if desc_submitted:
                        # Save text_area content to session_state
                        st.session_state[desc_key] = desc_input
                        desc_path = st.session_state.submitted_file_path.replace(".py", "_desc.txt")
                        with open(desc_path, 'w', encoding='utf-8') as f:
                            f.write(desc_input)
                        st.success("Description saved! Host can view this.")
            else:
                st.info("Submit your code first to unlock the description box.")


    else:
        st.info("Enter a valid problem code (PS1..PS5) and click 'Open Problem'.")

# -------------- Organizer Notes --------------
# How to run:
#   1) Install dependencies on HOST machine: pip install streamlit scikit-learn numpy pandas
#   2) Put your logo file under assets/logo.png (or change LOGO_PATH)
#   3) Run the app: streamlit run streamlit_symposium_code_editor.py
#   4) All saved data lives in ./event_data on the HOST machine.
#      If multiple client machines open the app via network, their actions are stored centrally.
#
# How to customize problems:
#   - Edit the PROBLEMS dict above. For each entry, update 'starter' code and the validator function.
#   - Keep function names consistent between starter code and validator expectations.
#   - For dataset/model based problems, load your dataset at the top of the starter code and test it in the validator.
#   - You can tighten/loosen thresholds in validators (e.g., accuracy >= X).
#   - To forbid additional modules, edit ALLOWED_IMPORTS. The sanitizer blocks unknown imports and dangerous calls.
#
# Security notes:
#   - This uses Python's exec with a restricted builtins set and a basic AST sanitizer.
#   - It is NOT a perfect sandbox. For public/hosted events, consider containerization or separate worker processes.
#
# Submission policy:
#   - The app enforces one submission per team per problem by checking event_data/submissions.csv.
#   - Admin can review in the Admin page (password from EVENT_ADMIN_PASSWORD env var or default shown above).
