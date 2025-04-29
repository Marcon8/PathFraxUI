from django.shortcuts import render
from .ml_model import SimpleRFModel
import pandas as pd

# Instantiate the model using the dataset
rf_model = SimpleRFModel(dataset_path='/Users/mariacontreras/Desktop/PathFRAX/path_combine_final_outcome_vars 2.csv')

def index(request):
    context = {}  # Dictionary to hold data for the template
    if request.method == "POST" and request.FILES.get("csv_file"):
        csv_file = request.FILES["csv_file"]
        try:
            # Read the uploaded CSV file
            data = pd.read_csv(csv_file)

            # Exclude the 'fracture' column (target variable) if present
            if 'fracture' in data.columns:
                data = data.drop(columns=['fracture'])

            # Validate that the uploaded file has the required features
            missing_features = [feature for feature in rf_model.X.columns if feature not in data.columns]
            if missing_features:
                context['error'] = f"Missing required features: {', '.join(missing_features)}"
                return render(request, "pathfrax/index.html", context)

            # Select the first patient's data (first row)
            first_patient_data = data.iloc[0:1]  # Select the first row as a DataFrame
            
            # Predict using the model
            prediction = rf_model.predict(first_patient_data)[0]  # Get the first prediction
            
            # Map 'N' and 'P' to human-readable messages
            if prediction == 'N':
                context['prediction'] = "Not likely to fracture"
            elif prediction == 'P':
                context['prediction'] = "Likely to fracture"
        except Exception as e:
            context['error'] = str(e)

    # Render the page with context
    return render(request, "pathfrax/index.html", context)
