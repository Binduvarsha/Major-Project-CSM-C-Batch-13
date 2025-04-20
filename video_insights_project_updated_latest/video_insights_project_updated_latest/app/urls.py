from django.urls import path
from app import views as views

urlpatterns = [
    path('about/',views.about,name="about"),
    path('login/',views.user_login,name="user_login"),
    path('register/',views.user_register,name="user_register"),
    path('contact/',views.contact,name="contact"),
    path('otp/',views.user_otp,name="user_otp"),
    path('contact/',views.contact,name="contact"),
    path('logout/', views.logout, name='logout'),
    path("dashboard/", views.user_dashboard,name="user_dashboard"),
    path('user/profile/',views.user_profile,name="user_profile"),
    path('user/feedback/',views.user_feedbacks,name="user_feedbacks"),



    path('upload/', views.upload_video, name='upload_video'),  
    path('video/<int:pk>/', views.video_detail, name='video_detail'),  
    path('api/ask-question/', views.ask_question, name='ask_question'),
    path('api/summarize-text/', views.summarize_text, name='summarize_text'),
    path('translate/',views.translate_text, name='translate_text'),
    path('detection/history/',views.detection_history, name='detection_history'),
    path('translate_video/<int:video_id>/', views.translate_video, name='translate_video'),
    path('ask_question/<int:video_id>/', views.ask_question, name='ask_question'),
    path('summarize_video/<int:video_id>/', views.summarize_video, name='summarize_video'),
    path('play-video/<int:pk>/', views.play_video, name='play_video'),
    path('play-audio/<int:pk>/', views.play_audio, name='play_audio'),
    path('detect_themes/<int:pk>/', views.detect_themes, name='detect_themes'),
    path('detect_themes_model/<int:pk>/', views.detect_themes_model, name='detect_themes_model'),





]