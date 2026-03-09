import pandas as pd
from django.shortcuts import render
from predictor.data_exploration import dataset_exploration, data_exploration, rwanda_vehicle_map
import joblib
from model_generators.clustering.train_cluster import evaluate_clustering_model
from model_generators.classification.train_classifier import evaluate_classification_model
from model_generators.regression.train_regression import evaluate_regression_model
import numpy as np

# Load models once
regression_model = joblib.load(
    "model_generators/regression/regression_model.pkl")
classification_model = joblib.load(
    "model_generators/classification/classification_model.pkl")
clustering_model = joblib.load(
    "model_generators/clustering/clustering_model.pkl")


def data_exploration_view(request):
    df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
    
    # Calculate coefficient of variation for clustering
    def calculate_coefficient_of_variation(df):
        # Use income and selling price for CV calculation
        income_cv = df['estimated_income'].std() / df['estimated_income'].mean()
        price_cv = df['selling_price'].std() / df['selling_price'].mean()
        return {
            'income_cv': round(income_cv * 100, 2),
            'price_cv': round(price_cv * 100, 2),
            'average_cv': round(((income_cv + price_cv) / 2) * 100, 2)
        }
    
    # Get clustering evaluation and add CV
    clustering_eval = evaluate_clustering_model()
    cv_metrics = calculate_coefficient_of_variation(df)
    
    context = {
        "data_exploration": data_exploration(df),
        "dataset_exploration": dataset_exploration(df),
        "rwanda_map": rwanda_vehicle_map(df),
        "cv_metrics": cv_metrics,
        "evaluations": {
            "regression": evaluate_regression_model(),
            "classification": evaluate_classification_model(),
            "clustering": clustering_eval
        }
    }
    
    # Handle form submissions for all models
    if request.method == "POST":
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])
        
        # Determine which form was submitted based on button name or hidden input
        form_type = request.POST.get("form_type", "")
        
        if form_type == "regression" or "Predict Market Price" in request.POST.get("submit", ""):
            prediction = regression_model.predict([[year, km, seats, income]])[0]
            context["price"] = prediction
            
        elif form_type == "classification" or "Predict Income Category" in request.POST.get("submit", ""):
            prediction = classification_model.predict([[year, km, seats, income]])[0]
            context["prediction"] = prediction
            
        elif form_type == "clustering" or "Run Combined Inference" in request.POST.get("submit", ""):
            # Step 1: Predict price
            predicted_price = regression_model.predict([[year, km, seats, income]])[0]
            # Step 2: Predict cluster
            cluster_id = clustering_model.predict([[income, predicted_price]])[0]
            mapping = {
                0: "Economy",
                1: "Standard",
                2: "Premium"
            }
            context.update({
                "prediction": mapping.get(cluster_id, "Unknown"),
                "price": predicted_price
            })
    
    return render(request, "predictor/index.html", context)
