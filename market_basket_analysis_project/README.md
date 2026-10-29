# Market Basket Analysis — Data Mining Course Project

Interactive Streamlit dashboard for Association Rule Mining.

## Features
- Built-in sample transactions, so the app can run immediately
- Upload your own transaction CSV
- Apriori and FP-Growth algorithms
- Support, confidence, and lift thresholds
- Product frequency chart, frequent-itemset chart, association-rules table, and interactive scatter plot
- Download generated association rules as CSV

## 1. Requirements
Install Python 3.11 or newer from https://www.python.org/downloads/ . During setup, enable **Add Python to PATH**.

## 2. Open the project folder
Open Command Prompt or PowerShell in this folder.

## 3. Create and activate a virtual environment (Windows)
```bat
python -m venv .venv
.venv\Scripts\activate
```
If PowerShell blocks activation, use Command Prompt and run the two commands there.

## 4. Install packages
```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Run the dashboard
```bat
python -m streamlit run app.py
```
The dashboard should open in your browser, usually at http://localhost:8501.

## CSV upload format
Use one row per product in a transaction. For example:

```csv
Transaction,Item
T001,Bread
T001,Milk
T001,Butter
T002,Bread
T002,Milk
T003,Coffee
T003,Sugar
```

The column names must match the sidebar fields (default: `Transaction` and `Item`).

## Suggested real dataset
The UCI Online Retail dataset contains invoice-level transaction data:
https://archive.ics.uci.edu/dataset/352/online+retail
Use `InvoiceNo` as the transaction ID and `Description` as the product. Remove missing values and cancellation invoices (invoice numbers starting with C), and consider limiting to a manageable subset before running algorithms.

## GitHub upload
1. Create a new repository on https://github.com/new .
2. Name it `market-basket-analysis`.
3. In the project folder, upload `app.py`, `requirements.txt`, and `README.md` (do not upload `.venv`).
4. Commit the files.
5. To deploy later, use Streamlit Community Cloud and select the GitHub repository and `app.py`.

## Project explanation
- Support: share of baskets containing an itemset.
- Confidence: conditional frequency of the consequent when the antecedent occurs.
- Lift: observed co-occurrence divided by the co-occurrence expected under independence.
