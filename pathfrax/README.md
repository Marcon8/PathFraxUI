
PathFRAX Web Application

This is a Django-based web application designed to predict the likelihood of bone fracture using a pre-trained Random Forest model. The prediction is based on input data provided by users in CSV format.

Purpose:
The application allows users (e.g., clinicians or researchers) to upload a CSV file containing patient data. It processes the file, validates required features, and predicts whether the first patient listed is likely to experience a fracture.


Features:
- Accepts CSV file uploads through a web interface.
- Validates input features against the model's expectations.
- Returns a prediction: "Likely to fracture" or "Not likely to fracture".
- Displays errors if required features are missing or the input is invalid.

How to Run the Application: 
1. Clone the repository and navigate into the project directory.
2. Install dependencies (see below).
3. Ensure your `ml_model.py` file and the trained model with the dataset path are correctly placed.
4. Update dataset path in `views.py` if needed:
   rf_model = SimpleRFModel(dataset_path='/absolute/path/to/dataset.csv')
5. Start the Django development server:
   python manage.py runserver
6. Visit http://localhost:8000 in your browser.

File Structure
- views.py: Contains logic for processing uploads and making predictions.
- ml_model.py: Expected to define the SimpleRFModel class used for inference.
- pathfrax/index.html: Template for the upload interface and result display.

 Dependencies
Make sure the following Python packages are installed:
- Django
- pandas
- scikit-learn

Install them using pip:
    pip install django pandas scikit-learn

CSV Input Requirements
- Must contain all the features used by the model (SimpleRFModel.X.columns).
- Should not include the target column 'fracture', as it is excluded during inference.
- Only the first row of the CSV will be used for prediction.
