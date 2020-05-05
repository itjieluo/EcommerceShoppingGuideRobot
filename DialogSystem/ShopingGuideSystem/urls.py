from django.urls import path
from ShopingGuideSystem import views

urlpatterns = [
    path(r'', views.index),
    path(r'response/', views.Dialog_sys)
]
