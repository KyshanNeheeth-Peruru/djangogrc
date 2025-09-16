from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import json
import os
import pandas as pd
from django.http import JsonResponse
from difflib import SequenceMatcher
import platform
from pdf2image import convert_from_bytes
from django.core.files.uploadedfile import UploadedFile
from PIL import Image
import io
import re
import base64
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from django.shortcuts import render
from dotenv import load_dotenv
from .models import Prompt

def get_prompt_text(name="default"):
    try:
        return Prompt.objects.get(name=name).content
    except Prompt.DoesNotExist:
        return "Default fallback prompt."


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
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


def brd_generation_view(request):
    context = {}

    if request.method == "POST" and request.FILES.get("brd_pdf"):
      # uploaded_file = request.FILES["brd_pdf"]
        
# =========================================================================================================
      uploaded_file = request.FILES["brd_pdf"]
      # Convert PDF to images (each page = one image)
      try:
        if platform.system() == "Windows":
          images = convert_from_bytes(
          uploaded_file.read(),
          dpi=200,
          poppler_path=r"C:\Users\kisha\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin"
          )
        else:
          images = convert_from_bytes(uploaded_file.read(), dpi=200)
      except Exception as e:
        context["error"] = f"Failed to convert PDF: {e}"
        return render(request, "brd_gen.html", context)

      complete_content = ""
      for i, image in enumerate(images):
        print(f"Processing page {i+1}...")
        extracted_text = call_openai_vision(image, i+1)
        complete_content += f"\n\n--- Page {i+1} ---\n{extracted_text}"
      
      print("========================================================================")
      # print(complete_content)
      requirements = extract_structured_requirements(complete_content)

      
        
# =========================================================================================================

      # requirements = [
      #       {"section": "1","id": "REQ-001", "description": "User login functionality", "type": "Functional"},
      #       {"section": "2","id": "REQ-002", "description": "Password encryption", "type": "Security"},
      #       {"section": "3","id": "REQ-003", "description": "Data backup every 24 hours", "type": "Non-Functional"},
      # ]
      
      print(requirements)
      
      sections = extract_brd_sections(complete_content)
      
      print(sections)

      context.update({
          "uploaded_file_name": uploaded_file.name,
          "requirements": requirements,
          "executive_summary": sections.get("Executive Summary", ""),
          "objectives": sections.get("Objectives", ""),
          "scope": sections.get("Scope", ""),
          "assumptions": sections.get("Assumptions", ""),
          "out_of_scope": sections.get("Out of Scope", ""),
          "stakeholders": sections.get("Stakeholders", "")
      })

      # context["requirements"] = requirements
      # context["uploaded_file_name"] = uploaded_file.name

    return render(request, "brd_gen.html", context)

def format_description_for_mathjax(desc: str) -> str:
    # 1. Clean symbols
    desc = desc.replace("×", r"\times")
    desc = desc.replace("²", "^2")
    desc = desc.replace("−", "-")
    desc = desc.replace("∑", r"\sum")
    desc = desc.replace("ρ", r"\rho")
    desc = desc.replace("γ", r"\gamma")
    desc = desc.replace("Δ", r"\Delta")
    desc = desc.replace("≤", r"\leq")
    desc = desc.replace("≥", r"\geq")

    # 2. Find likely formulas to wrap
    # These patterns assume formulas are standalone or comma-separated
    formula_pattern = re.compile(
        r'(?<!\\)(Kb\s*=.*?)(?=[.,;]|$)|'     # Match formula starting with Kb = ...
        r'(?<!\\)(WSk\s*=.*?)(?=[.,;]|$)|'    # Match WSk = ...
        r'(?<!\\)(Sb\s*=.*?)(?=[.,;]|$)|'     # Match Sb = ...
        r'(?<!\\)(Sc\s*=.*?)(?=[.,;]|$)|'     # Match Sc = ...
        r'(?<!\\)(CVRk[+-]\s*=.*?)(?=[.,;]|$)'  # Match CVRk+ = ...
    )

    def wrap_formula(match):
        formula = match.group(0)
        return r"\(" + formula.strip() + r"\)"

    # Apply the wrapping only on formula fragments
    desc = formula_pattern.sub(wrap_formula, desc)

    # Optional: turn newlines into <br> to preserve formatting
    desc = desc.replace("\n", "<br>")
    return desc



def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def call_openai_vision(image, page_number):
  base64_image = image_to_base64(image)
  
  prompt = get_prompt_text(name="desc_image")
  
  response = openai.ChatCompletion.create(model="gpt-4o",
    messages=[
      {"role": "user", "content": [{"type": "text", "text": prompt},
      {"type": "image_url", "image_url": {
      "url": f"data:image/png;base64,{base64_image}"}}]}
      ],max_tokens=2000,
    )

  return response.choices[0].message.content


def extract_structured_requirements(complete_content: str) -> list[dict]:
  prompt_text = get_prompt_text(name="brd_gen")
  prompt = prompt_text.format(complete_content= complete_content)

  response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=2000
    )

  raw_output = response.choices[0].message.content.strip()
  cleaned_output = clean_llm_response(raw_output)
  print(cleaned_output)
  return json.loads(cleaned_output)



def extract_brd_sections(document_text: str) -> dict:
  prompt_text = get_prompt_text(name="summary_gen")
  prompt = prompt_text.format(document_text= document_text)

  try:
    response = openai.ChatCompletion.create(
    model="gpt-4-1106-preview",
    messages=[
      {"role": "system", "content": "You extract structured sections from BRD documents and respond in JSON only."},
      {"role": "user", "content": prompt}
      ],temperature=0.2)
    content = response.choices[0].message.content.strip()
    try:
      return json.loads(content)
    except json.JSONDecodeError:
      json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
        print("Failed to parse JSON from LLM response.")
    return {
      "Executive Summary": "",
      "Objectives": "",
      "Scope": "",
      "Out of Scope": ""
    }

  except Exception as e:
    print("Error extracting BRD sections:", e)
    return {
      "Executive Summary": "",
      "Objectives": "",
      "Scope": "",
      "Out of Scope": ""
    }


def clean_llm_response(raw_output: str) -> str:
    """
    Remove markdown code block markers like ```python or ```
    """
    lines = raw_output.strip().splitlines()

    # Remove starting line if it's ``` or ```python
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]

    # Remove ending line if it's ```
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines).strip()

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
