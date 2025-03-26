from django.urls import path
from .views import search, search_rules, get_form_details

urlpatterns = [
    path("search/", search),
    path('search_rules/', search_rules, name='search_rules'),
    path("form_details/", get_form_details, name='get_form_details')

]
