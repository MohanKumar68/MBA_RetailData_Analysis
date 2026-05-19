import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from model_training import generate_and_train

st.set_page_config(page_title="Market Basket Analysis", layout="wide")

st.title("🛒 Market Basket Analysis using Apriori and FP-Growth")
st.write("Analyze Indian grocery transactions and get store layout optimization suggestions.")

if st.button("Generate Dataset and Train Models"):
    with st.spinner("Training models..."):
        results = generate_and_train()
    st.success("Training completed!")
    st.session_state["results"] = results

if "results" in st.session_state:
    results = st.session_state["results"]

    st.subheader("⏱️ Performance Comparison")
    perf = pd.DataFrame({
        "Algorithm": ["Apriori", "FP-Growth"],
        "Execution Time (s)": [results["apriori_time"], results["fpgrowth_time"]]
    })
    st.dataframe(perf, use_container_width=True)

    fig, ax = plt.subplots()
    ax.bar(perf["Algorithm"], perf["Execution Time (s)"])
    st.pyplot(fig)

    st.subheader("📌 Top Association Rules")
    st.dataframe(results["rules"].head(20), use_container_width=True)

    st.subheader("🏪 Store Layout Optimization Suggestions")
    for i, row in results["rules"].head(10).iterrows():
        ant = ", ".join(list(row["antecedents"]))
        cons = ", ".join(list(row["consequents"]))
        st.write(f"Place **{ant}** near **{cons}** (Confidence: {row['confidence']:.2f})")

    st.subheader("📊 Top Product Frequencies")
    st.bar_chart(results["item_support"].set_index("item")["support"])
else:
    st.info("Click the button above to generate data and train models.")
# =========================
# 🎯 Cross-Marketing Strategies
# =========================
st.subheader("🎯 Cross-Marketing Strategies")

rules_df = results["rules"].copy()

if not rules_df.empty:
    # Convert antecedents and consequents to strings
    rules_df["Trigger Product"] = rules_df["antecedents"].apply(lambda x: ", ".join(x))
    rules_df["Recommended Product"] = rules_df["consequents"].apply(lambda x: ", ".join(x))

    # Unique trigger products for dropdown
    trigger_products = sorted(rules_df["Trigger Product"].unique())

    # Dropdown to select a trigger product
    selected_trigger = st.selectbox(
        "Select a product to see cross-marketing recommendations:",
        trigger_products
    )

    # Filter recommendations for selected trigger
    selected_rules = rules_df[rules_df["Trigger Product"] == selected_trigger]

    if not selected_rules.empty:
        for _, row in selected_rules.head(5).iterrows():
            recommended = row["Recommended Product"]
            confidence = row["confidence"]
            lift = row["lift"]

            # Generate marketing strategy
            strategy = (
                f"Offer a combo discount for **{selected_trigger} + {recommended}**, "
                f"display **{recommended}** as 'Customers also bought this', "
                f"and send personalized promotional offers."
            )

            # Display in a nice box
            st.success(
                f"### 🛍 Recommended Product: {recommended}\n\n"
                f"**Confidence:** {confidence:.2f}\n\n"
                f"**Lift:** {lift:.2f}\n\n"
                f"**Strategy:** {strategy}"
            )
else:
    st.warning("No association rules available for cross-marketing recommendations.")
