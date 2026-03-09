from django.urls import path
from predictor import views

urlpatterns = [
    path("", views.data_exploration_view, name="home"),
]
