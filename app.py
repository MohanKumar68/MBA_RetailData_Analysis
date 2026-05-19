import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model_training import generate_and_train

st.set_page_config(page_title="SmartShelf", layout="wide")

st.title("🛒 SmartShelf")
st.write("Analyze Indian grocery transactions and get store layout optimization suggestions.")

# ==========================================================
# Generate dataset and train models
# ==========================================================
if st.button("Run Market Basket Analysis"):
    with st.spinner("Training models..."):
        st.session_state["results"] = generate_and_train()
    st.success("Training completed!")

# ==========================================================
# If results are NOT available, stop execution here.
# This prevents ALL NameError issues related to 'results'.
# ==========================================================
if "results" not in st.session_state:
    st.info("Click the button above to generate data and train models.")
    st.stop()

# Safe to use results after this point
results = st.session_state["results"]

# ==========================================================
# ⏱️ Performance Comparison
# ==========================================================
st.subheader("⏱️ Performance Comparison")

perf = pd.DataFrame({
    "Algorithm": ["Apriori", "FP-Growth"],
    "Execution Time (s)": [
        results["apriori_time"],
        results["fpgrowth_time"]
    ]
})

st.dataframe(perf, use_container_width=True)

fig, ax = plt.subplots()
ax.bar(perf["Algorithm"], perf["Execution Time (s)"])
st.pyplot(fig)

# ==========================================================
# 📌 Top Association Rules
# ==========================================================
st.subheader("📌 Top Association Rules")
st.dataframe(results["rules"].head(20), use_container_width=True)

# ==========================================================
# 🏪 Store Layout Optimization Suggestions
# ==========================================================
st.subheader("🏪 Store Layout Optimization Suggestions")

for _, row in results["rules"].head(10).iterrows():
    ant = ", ".join(list(row["antecedents"]))
    cons = ", ".join(list(row["consequents"]))
    st.write(
        f"Place **{ant}** near **{cons}** "
        f"(Confidence: {row['confidence']:.2f})"
    )

# ==========================================================
# 📊 Top Product Frequencies
# ==========================================================
st.subheader("📊 Top Product Frequencies")
st.bar_chart(
    results["item_support"].set_index("item")["support"]
)

# ==========================================================
# 🎯 Cross-Marketing Strategies
# ==========================================================
st.subheader("🎯 Cross-Marketing Strategies")

rules_df = results["rules"].copy()

if not rules_df.empty:
    # Convert itemsets to readable strings
    rules_df["Trigger Product"] = rules_df["antecedents"].apply(
        lambda x: ", ".join(x)
    )
    rules_df["Recommended Product"] = rules_df["consequents"].apply(
        lambda x: ", ".join(x)
    )

    # Dropdown list
    trigger_products = sorted(
        rules_df["Trigger Product"].unique()
    )

    selected_trigger = st.selectbox(
        "Select a product to see cross-marketing recommendations:",
        trigger_products
    )

    # Filter selected rules
    selected_rules = rules_df[
        rules_df["Trigger Product"] == selected_trigger
    ]

    if not selected_rules.empty:
        for _, row in selected_rules.head(5).iterrows():
            recommended = row["Recommended Product"]
            confidence = row["confidence"]
            lift = row["lift"]

            strategy = (
                f"Offer a combo discount for "
                f"**{selected_trigger} + {recommended}**, "
                f"display **{recommended}** as "
                f"'Customers also bought this', "
                f"and send personalized promotional offers."
            )

            st.success(
                f"### 🛍 Recommended Product: {recommended}\n\n"
                f"**Confidence:** {confidence:.2f}\n\n"
                f"**Lift:** {lift:.2f}\n\n"
                f"**Strategy:** {strategy}"
            )
else:
    st.warning(
        "No association rules available for cross-marketing recommendations."
    )
