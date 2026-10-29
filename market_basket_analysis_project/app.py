
import streamlit as st
import pandas as pd
import plotly.express as px
import time
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Market Basket Analysis")
st.caption(
    "Discover products purchased together using Association Rule Mining"
)

SAMPLE = [
    ["Bread", "Milk", "Butter"],
    ["Bread", "Milk"],
    ["Bread", "Butter"],
    ["Milk", "Butter"],
    ["Bread", "Milk", "Butter"],
    ["Eggs", "Bread"],
    ["Eggs", "Milk"],
    ["Bread", "Milk", "Eggs"],
    ["Coffee", "Sugar"],
    ["Coffee", "Milk", "Sugar"],
    ["Tea", "Sugar"],
    ["Coffee", "Sugar"],
    ["Bread", "Jam"],
    ["Bread", "Milk", "Jam"],
    ["Milk", "Cereal"],
    ["Cereal", "Milk"],
    ["Eggs", "Bread", "Milk"],
    ["Tea", "Biscuits"],
    ["Tea", "Sugar", "Biscuits"],
    ["Coffee", "Biscuits"]
]

# ---------------- SIDEBAR SETTINGS ----------------

with st.sidebar:
    st.header("⚙️ Settings")

    uploaded = st.file_uploader(
        "Upload transaction file",
        type=["csv", "xlsx", "xls"],
        help="Upload a CSV or Excel retail dataset."
    )

    st.markdown(
        "**UCI Online Retail column names:** "
        "`InvoiceNo` and `Description`."
    )

    id_col = st.text_input(
        "Transaction ID column",
        value="InvoiceNo"
    )

    item_col = st.text_input(
        "Product/item column",
        value="Description"
    )

    algorithm = st.selectbox(
        "Algorithm",
        ["Apriori", "FP-Growth"]
    )

    min_support = st.slider(
        "Minimum support",
        0.01, 0.50, 0.03, 0.01
    )

    min_confidence = st.slider(
        "Minimum confidence",
        0.10, 1.00, 0.40, 0.05
    )

    min_lift = st.slider(
        "Minimum lift",
        0.5, 5.0, 1.0, 0.1
    )

    max_transactions = st.number_input(
        "Maximum transactions",
        min_value=100,
        max_value=50000,
        value=5000,
        step=500
    )

    max_products = st.number_input(
        "Maximum products",
        min_value=20,
        max_value=1000,
        value=100,
        step=20
    )

# ---------------- LOAD DATA ----------------

raw = None
source_name = "Built-in sample transactions"

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith((".xlsx", ".xls")):
            raw = pd.read_excel(uploaded)
        else:
            raw = pd.read_csv(uploaded)

        raw.columns = [str(c).strip() for c in raw.columns]
        source_name = uploaded.name

    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        st.stop()

if raw is None:
    transactions = SAMPLE

else:
    if id_col not in raw.columns or item_col not in raw.columns:
        st.error(
            f"Required columns not found: '{id_col}' and '{item_col}'. "
            f"Available columns: {', '.join(raw.columns)}"
        )
        st.stop()

    # Remove rows with missing invoice or product
    raw = raw.dropna(
        subset=[id_col, item_col]
    ).copy()

    raw[id_col] = raw[id_col].astype(str).str.strip()
    raw[item_col] = raw[item_col].astype(str).str.strip()

    # Remove cancelled invoices
    raw = raw[
        ~raw[id_col].str.upper().str.startswith("C")
    ].copy()

    # Remove returns and invalid quantities
    if "Quantity" in raw.columns:
        raw["Quantity"] = pd.to_numeric(
            raw["Quantity"],
            errors="coerce"
        )
        raw = raw[raw["Quantity"] > 0].copy()

    # Remove empty values
    raw = raw[
        (raw[id_col] != "") &
        (raw[item_col] != "")
    ].copy()

    if raw.empty:
        st.warning("No valid transactions remain after cleaning.")
        st.stop()

    # Limit invoices to protect laptop memory
    invoice_ids = (
        raw[id_col]
        .drop_duplicates()
        .head(int(max_transactions))
    )

    raw = raw[raw[id_col].isin(invoice_ids)].copy()

    # Keep the most common products
    common_items = (
        raw[item_col]
        .value_counts()
        .head(int(max_products))
        .index
    )

    raw = raw[raw[item_col].isin(common_items)].copy()

    # Group products into shopping baskets
    transactions = (
        raw.groupby(id_col, sort=False)[item_col]
        .apply(lambda values: sorted(set(values)))
        .tolist()
    )

    transactions = [
        items for items in transactions if items
    ]

if not transactions:
    st.warning("No transactions available.")
    st.stop()

# ---------------- ENCODE TRANSACTIONS ----------------

encoder = TransactionEncoder()

encoded = encoder.fit(transactions).transform(transactions)

basket = pd.DataFrame(
    encoded,
    columns=encoder.columns_,
    dtype=bool
)

# ---------------- DATA MINING ----------------

with st.spinner(f"Running {algorithm}..."):
    started = time.perf_counter()

    try:
        if algorithm == "Apriori":
            itemsets = apriori(
                basket,
                min_support=float(min_support),
                use_colnames=True,
                max_len=3
            )
        else:
            itemsets = fpgrowth(
                basket,
                min_support=float(min_support),
                use_colnames=True,
                max_len=3
            )

        if itemsets.empty:
            rules = pd.DataFrame()

        else:
            rules = association_rules(
                itemsets,
                metric="confidence",
                min_threshold=float(min_confidence)
            )

            if not rules.empty:
                rules = rules[
                    rules["lift"] >= float(min_lift)
                ].copy()

                rules["Antecedents"] = rules[
                    "antecedents"
                ].apply(
                    lambda x: ", ".join(sorted(x))
                )

                rules["Consequents"] = rules[
                    "consequents"
                ].apply(
                    lambda x: ", ".join(sorted(x))
                )

                rules = rules.sort_values(
                    ["lift", "confidence"],
                    ascending=False
                )

        elapsed = time.perf_counter() - started

    except Exception as exc:
        st.error(f"Data mining failed: {exc}")
        st.info(
            "Try reducing the transaction/product limits "
            "or increasing minimum support."
        )
        st.stop()

# ---------------- DASHBOARD METRICS ----------------

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Transactions analyzed",
    f"{len(transactions):,}"
)

c2.metric(
    "Unique products",
    f"{len(encoder.columns_):,}"
)

c3.metric(
    "Frequent itemsets",
    f"{len(itemsets):,}"
)

c4.metric(
    "Association rules",
    f"{len(rules):,}"
)

st.info(
    f"Data source: {source_name} | "
    f"Algorithm: {algorithm} | "
    f"Mining time: {elapsed:.2f} seconds"
)

# ---------------- PRODUCT FREQUENCY CHART ----------------

left, right = st.columns(2)

with left:
    st.subheader("📊 Most Frequently Purchased Products")

    counts = pd.Series([
        item
        for transaction in transactions
        for item in set(transaction)
    ]).value_counts().head(12)

    product_df = counts.rename_axis(
        "Product"
    ).reset_index(
        name="Transactions containing product"
    )

    fig = px.bar(
        product_df,
        x="Product",
        y="Transactions containing product",
        title="Top Products by Basket Count"
    )

    fig.update_layout(xaxis_tickangle=-35)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------- FREQUENT ITEMSETS CHART ----------------

with right:
    st.subheader("🧺 Frequent Itemsets")

    if itemsets.empty:
        st.warning(
            "No frequent itemsets found. "
            "Lower the minimum support."
        )

    else:
        view = itemsets.copy()

        view["Items"] = view["itemsets"].apply(
            lambda x: ", ".join(sorted(x))
        )

        view = view.sort_values(
            "support",
            ascending=False
        ).head(15)

        fig = px.bar(
            view,
            x="support",
            y="Items",
            orientation="h",
            title="Frequent Itemsets by Support"
        )

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"}
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ---------------- ASSOCIATION RULES ----------------

st.subheader("🔗 Association Rules")

if rules.empty:
    st.warning(
        "No rules match your thresholds. "
        "Try lowering confidence or lift."
    )

else:
    display_cols = [
        "Antecedents",
        "Consequents",
        "support",
        "confidence",
        "lift"
    ]

    st.dataframe(
        rules[display_cols].round(3),
        use_container_width=True,
        hide_index=True
    )

    chart_rules = rules.head(20).copy()

    chart_rules["Rule"] = (
        chart_rules["Antecedents"] +
        " → " +
        chart_rules["Consequents"]
    )

    fig = px.scatter(
        chart_rules,
        x="confidence",
        y="lift",
        size="support",
        hover_name="Rule",
        title="Confidence vs Lift",
        size_max=30
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    csv_bytes = rules[display_cols].to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Association Rules",
        data=csv_bytes,
        file_name="association_rules.csv",
        mime="text/csv"
    )

# ---------------- METRIC EXPLANATION ----------------

with st.expander("📘 Understand the Metrics"):
    st.markdown("""
    **Support:** Proportion of transactions containing an itemset.

    **Confidence:** Proportion of transactions containing the
    antecedent that also contain the consequent.

    **Lift:** Observed co-occurrence divided by the co-occurrence
    expected if the items were independent.

    **Lift greater than 1:** Indicates a positive association
    in this dataset. It does not prove causation.
    """)

st.caption(
    "Market Basket Analysis | Data Mining Course Project"
)
