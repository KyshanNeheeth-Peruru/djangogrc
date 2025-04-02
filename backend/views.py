from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import json
import os
import pandas as pd
from django.http import JsonResponse
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_FILE = os.path.join(ROOT_DIR, "datav2.json")
EXCEL_FILE_PATH = os.path.join(ROOT_DIR, "Lineagev2.xlsx")
model = SentenceTransformer('all-MiniLM-L6-v2')


@api_view(["GET"])
def search(request):
    query = request.GET.get("query", "").lower()

    if not query:
        return Response({"error": "Query parameter is required"}, status=400)

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        return Response({"error": "Data file not found"}, status=500)
    
    def calculate_similarity(a, b):
        embeddings = model.encode([a, b])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return f"{round(similarity * 100)}%"

    # def calculate_similarity(a, b):
    #     ratio = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    #     return f"{round(ratio * 100)}%"

    results = []

    for item in data:
        if query in item["content"].lower():
            new_metadata = {}
            for form_key, form_entries in item["metadata"].items():
                updated_entries = []
                for entry in form_entries:
                    entry["similarity_index"] = calculate_similarity(query, entry["line"])
                    updated_entries.append(entry)
                new_metadata[form_key] = updated_entries

            results.append({
                "content": item["content"],
                "metadata": new_metadata
            })

    return Response({"results": results if results else "No results found"})


@api_view(["GET"])
def search_rules(request):
    value = request.GET.get("value", "").strip()
    if not value:
        return Response({"error": "Value parameter is required"}, status=400)

    try:
        df = pd.read_excel(EXCEL_FILE_PATH, engine="openpyxl")
    except FileNotFoundError:
        return Response({"error": "Excel file not found"}, status=500)
    except Exception as e:
        return Response({"error": str(e)}, status=500)
    filtered_rows = df[df["value"].astype(str) == value]
    results = [
        {
            "content": row["Line Item ID"],
            "metadata": {
                "Region": row["Region"],
                "Country": row["Country"],
                "Form Name": row["Form Name"],
                "Schedule Name": row["Schedule Name"],
                "Regulatory Code": row["Regulatory Code"],
                "Rule ID": row["RULE ID"],
                "SQL": row["SQL"],
                "Staging Transformation Rule": row["Staging Transformation Rule"],
                "Data Mart Transformation": row["Data Mart Transformation"],
                "TRL Transformation": row["TRL Transformation"],
            },
        }
        for _, row in filtered_rows.iterrows()
    ]

    return Response({"results": results if results else "No results found"})

@api_view(["GET"])
def get_form_details(request):
    form_name = request.GET.get("form_name", "").strip().lower()
    date = request.GET.get("date", "").strip().lower()
    country = request.GET.get("country", "").strip().lower()

    if not form_name or not date or not country:
        return Response({"error": "form_name, date, and country are required"}, status=400)

    try:
        with open("form_details.json", "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        return Response({"error": "Data file not found"}, status=500)

    for item in data:
        if (
            item["form_name"].lower() == form_name
            and item["date"].lower() == date
            and item["country"].lower() == country
        ):
            return Response(item)

    return Response({"error": "No matching record found"}, status=404)
