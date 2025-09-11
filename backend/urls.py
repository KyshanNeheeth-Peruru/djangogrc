from django.urls import path
from .views import search, search_rules, get_form_details, lineage_graph_view, brd_generation_view

urlpatterns = [
    path("search/", search),
    path('search_rules/', search_rules, name='search_rules'),
    path("form_details/", get_form_details, name='get_form_details'),
    path('lineage/', lineage_graph_view, name='lineage-graph'),
    path("brd_generation/", brd_generation_view, name = 'brd_generation'),


]
