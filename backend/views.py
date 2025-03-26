from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import json
import os
import pandas as pd
from django.http import JsonResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_FILE = os.path.join(ROOT_DIR, "datav2.json")
EXCEL_FILE_PATH = os.path.join(ROOT_DIR, "Lineagev2.xlsx")

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

    results = [
        {"content": item["content"], "metadata": item["metadata"]}
        for item in data if query in item["content"].lower()
    ]

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
