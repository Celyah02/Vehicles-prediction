from django.urls import path
from predictor import views

urlpatterns = [
    path("", views.data_exploration_view, name="home"),
    path("regression/", views.regression_analysis_view, name="regression_analysis"),
    path("classification/", views.classification_analysis_view, name="classification_analysis"),
    path("clustering/", views.clustering_analysis_view, name="clustering_analysis"),
]
