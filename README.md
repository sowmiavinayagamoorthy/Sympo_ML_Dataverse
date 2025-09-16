🧑‍💻 SYMPO – Machine Learning Streamlit App

This project is a Streamlit-based interactive application that allows users to learn, test, and experiment with basic Machine Learning models such as Logistic Regression and Naive Bayes. The app also includes validators,question prompts, and data validation to help users practice ML concepts in an interactive way.

🚀 Features

📊 Upload or use sample datasets.
⚙️ Train ML models like Logistic Regression and Naive Bayes.
📝 Validators for input fields (e.g., required features).
❓ Dynamic questions and quizzes for each problem statement.
🔍 Model evaluation with accuracy, confusion matrix, and classification reports.
🌐 Simple and interactive UI built with Streamlit.

Create a virtual environment::
     python -m venv venv
            venv\Scripts\activate   # On Windows
            source venv/bin/activate  # On Mac/Linux
Install the dependencies:: 
            pip install -r requirements.txt
▶️ Running the Application::
            Run the Streamlit app with:
                streamlit run sympo.py
            This will open the app in your default browser at:
                http://localhost:8501
📂 Project Structure::
           sympo-ml-app/
                │── sympo.py              # Main Streamlit application
                │── requirements.txt      # Dependencies
                │── README.md             # Project documentation
                │── sample_data.csv       # Example dataset 
📦 Dependencies::
           Main libraries used:
                Streamlit – for UI
                Pandas, NumPy – for data handling
                Scikit-learn – for ML models (Logistic Regression, Naive Bayes, etc.)

🧑‍🏫 Example Use Cases::
           Train a Logistic Regression model on health data (e.g., diabetes prediction).
           Try Naive Bayes for text or categorical problems.
           Validate data and handle missing values interactively.
           Learn ML concepts with simple built-in questions & quizzes.

               
             
