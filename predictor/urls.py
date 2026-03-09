from django.urls import path
from predictor import views

urlpatterns = [
    path("data_exploration", views.data_exploration_view, name="data_exploration"),
    path("regression_analysis", views.regression_analysis,
         name="regression_analysis"),
]
