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
from django.shortcuts import render

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_FILE = os.path.join(ROOT_DIR, "datav2.json")
EXCEL_FILE_PATH = os.path.join(ROOT_DIR, "Lineagev2.xlsx")
model = SentenceTransformer('all-MiniLM-L6-v2')

def lineage_graph_view(request):
    lineage_json = {
  "HC_C": {
    "tables": [""],
    "value_flows_from": ["Loan_Accounting_Subledger", "Core_Banking_Transactions", "Amortization_Schedule_Table"],
    "columns": {
      "CI_loans": [[""], [""]],
      "Real_estate_loans": [[""], [""]],
      "Consumer_loans": [[""], [""]],
      "Lease_financing_receivables": [[""], [""]],
      "Federal_funds_sold": [[""], [""]],
      "Unearned_income": [[""], [""]]
    },
    "table_name": "HC_C",
    "sql": "this is a base table from Schedule HC-C: Loans and Lease Financing Receivables"
  },
  "HC_N": {
    "tables": [""],
    "value_flows_from": ["Loan_Provisioning_Engine", "Allowance_Calculation_Table"],
    "columns": {
      "Interest_accruals": [[""], [""]],
      "Credit_impairment_reserve": [[""], [""]]
    },
    "table_name": "HC_N",
    "sql": "this is a base table from Schedule HC-N: Allowance for Loan and Lease Losses"
  },
  "HC_A": {
    "tables": [""],
    "value_flows_from": ["Loan_Sales_System", "Held_For_Sale_Tracker"],
    "columns": {
      "CI_loans_HFS": [[""], [""]],
      "Real_estate_loans_HFS": [[""], [""]],
      "Consumer_loans_HFS": [[""], [""]],
      "Foreign_loans_HFS": [[""], [""]]
    },
    "table_name": "HC_A",
    "sql": "this is a base table from Schedule HC-A: Loans and Leases Held for Sale"
  },
  "HC_Q": {
    "tables": [""],
    "value_flows_from": ["Fair_Value_Model_Results", "Market_Data_Feed", "Valuation_Adjustments_Table"],
    "columns": {
      "Fair_value_adjustments_HFS": [[""], [""]],
      "Fair_value_adjustments_AFS": [[""], [""]]
    },
    "table_name": "HC_Q",
    "sql": "this is a base table from Schedule HC-Q: Fair Value Measurements"
  },
  "HC_B": {
    "tables": [""],
    "value_flows_from": ["Securities_Inventory", "Investment_Subledger", "Trade_Execution_Reports"],
    "columns": {
      "Available_for_sale_debt_securities": [[""], [""]]
    },
    "table_name": "HC_B",
    "sql": "this is a base table from Schedule HC-B: Securities"
  },
  "HC_Q_assets": {
    "tables": ["HC_C", "HC_N", "HC_A", "HC_Q", "HC_B"],
    "value_flows_from": [],
    "columns": {
      "Loans_and_leases_held_for_sale": [
        ["HC_A.CI_loans_HFS", "HC_A.Real_estate_loans_HFS", "HC_A.Consumer_loans_HFS", "HC_A.Foreign_loans_HFS"],
        ["HC_Q.Fair_value_adjustments_HFS"]
      ],
      "Loans_and_leases_held_for_investment": [
        ["HC_C.CI_loans", "HC_C.Real_estate_loans", "HC_C.Consumer_loans", "HC_C.Lease_financing_receivables"],
        ["HC_C.Unearned_income", "HC_N.Interest_accruals"]
      ],
      "Federal_funds_sold": [
        ["HC_C.Federal_funds_sold"],
        ["Credit_impairment_reserve"]
      ],
      "Available_for_sale_debt_securities": [
        ["HC_B.Available_for_sale_debt_securities", "HC_N.Credit_impairment_reserve"],
        ["HC_Q.Fair_value_adjustments_AFS", "HC_N.Interest_accruals"]
      ]
    },
    "table_name": "HC_Q_assets",
    "sql": "SELECT derived asset balances including Loans Held for Investment/Sale and other components from HC_C, HC_N, HC_A, HC_Q, HC_B"
  },
  "Rule_lineage": {
    "tables": ["HC_Q_assets"],
    "columns": {
      "Transaction_ID": [
        ["HC_Q_assets.Loans_and_leases_held_for_sale"],
        ["HC_Q_assets.Loans_and_leases_held_for_investment", "HC_Q_assets.Federal_funds_sold"]
      ],
      "Amount": [
        ["HC_Q_assets.Loans_and_leases_held_for_investment"],
        ["HC_Q_assets.Loans_and_leases_held_for_investment", "HC_Q_assets.Federal_funds_sold"]
      ],
      "Counter_party": [
        ["HC_Q_assets.Available_for_sale_debt_securities"],
        ["HC_Q_assets.Loans_and_leases_held_for_investment", "HC_Q_assets.Federal_funds_sold"]
      ],
      "Securities_Purchased": [
        ["HC_Q_assets.Loans_and_leases_held_for_investment"],
        ["HC_Q_assets.Loans_and_leases_held_for_investment", "HC_Q_assets.Federal_funds_sold"]
      ],
      "Quarterly_investment": [
        [""],
        [""]
      ]
    },
    "table_name": "Rule_lineage",
    "sql": "SELECT Transaction_ID, Amount, Counterparty, CASE WHEN Amount < 1000000 THEN 'Below Threshold' ELSE 'Above Threshold' END AS Securities_Purchased FROM HC_Q_assets WHERE Transaction_Type = 'Federal Funds Sold' AND Amount < 1000000"
  }
}
    
    return render(request, 'lineage_graph.html', {"inline_data": json.dumps(lineage_json)})

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
