import pandas as pd
import random
import time
import os
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

PRODUCTS = [
    "Rice", "Atta", "Toor Dal", "Urad Dal", "Idli Rice", "Sambar Powder",
    "Tea Powder", "Biscuits", "Maggi", "Curd", "Paneer", "Ghee",
    "Jaggery", "Groundnut Oil", "Coconut Oil"
]

PATTERNS = [
    ["Rice", "Toor Dal", "Sambar Powder"],
    ["Atta", "Ghee"],
    ["Idli Rice", "Urad Dal"],
    ["Tea Powder", "Biscuits"],
    ["Curd", "Paneer"],
    ["Maggi", "Biscuits"],
]


def generate_transactions(n=1000):
    transactions = []

    for _ in range(n):
        basket = set(random.sample(PRODUCTS, random.randint(2, 5)))

        if random.random() < 0.7:
            basket.update(random.choice(PATTERNS))

        transactions.append(list(basket))

    return transactions


def save_transactions(transactions, path="data/transactions.csv"):

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convert transactions to tabular format
    max_len = max(len(t) for t in transactions)
    rows = [t + [None] * (max_len - len(t)) for t in transactions]

    # Save CSV
    pd.DataFrame(rows).to_csv(path, index=False)


def preprocess(transactions):

    # One-hot encoding
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)

    return pd.DataFrame(arr, columns=te.columns_)


def generate_and_train():

    # Generate transactions
    transactions = generate_transactions()

    # Save dataset
    save_transactions(transactions)

    # Preprocess dataset
    df = preprocess(transactions)

    # =========================
    # Apriori Algorithm
    # =========================
    start = time.time()

    freq_ap = apriori(
        df,
        min_support=0.05,
        use_colnames=True
    )

    apriori_time = time.time() - start

    # =========================
    # FP-Growth Algorithm
    # =========================
    start = time.time()

    freq_fp = fpgrowth(
        df,
        min_support=0.05,
        use_colnames=True
    )

    fpgrowth_time = time.time() - start

    # =========================
    # Association Rules
    # =========================
    rules = association_rules(
        freq_fp,
        metric="confidence",
        min_threshold=0.5
    )

    if not rules.empty:

        rules = rules.sort_values(
            ["confidence", "lift"],
            ascending=False
        )

        rules = rules[
            [
                "antecedents",
                "consequents",
                "support",
                "confidence",
                "lift"
            ]
        ]

        rules["antecedents"] = rules["antecedents"].apply(
            lambda x: sorted(list(x))
        )

        rules["consequents"] = rules["consequents"].apply(
            lambda x: sorted(list(x))
        )

    # =========================
    # Product Support Table
    # =========================
    item_support = pd.DataFrame({
        "item": df.columns,
        "support": df.mean().values
    }).sort_values(
        "support",
        ascending=False
    ).head(15)

    # ==========================================================
    # GRAPH GENERATION
    # ==========================================================

    # Create graphs directory
    os.makedirs("graphs", exist_ok=True)

    # ==========================================================
    # 1. Performance Comparison Graph
    # ==========================================================
    plt.figure(figsize=(6, 4))

    algorithms = ["Apriori", "FP-Growth"]
    times = [apriori_time, fpgrowth_time]

    plt.bar(algorithms, times)

    plt.title("Algorithm Performance Comparison")
    plt.ylabel("Execution Time (seconds)")
    plt.xlabel("Algorithm")

    plt.tight_layout()

    plt.savefig("graphs/performance_comparison.png")

    plt.close()

    # ==========================================================
    # 2. Top Frequent Products Graph
    # ==========================================================
    top_items = item_support.head(10)

    plt.figure(figsize=(8, 5))

    plt.bar(
        top_items["item"],
        top_items["support"]
    )

    plt.title("Top 10 Most Frequent Products")
    plt.ylabel("Support")
    plt.xlabel("Product")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig("graphs/top_frequent_products.png")

    plt.close()

    # ==========================================================
    # 3. Support vs Confidence Scatter Plot
    # ==========================================================
    if not rules.empty:

        plt.figure(figsize=(7, 5))

        plt.scatter(
            rules["support"],
            rules["confidence"],
            s=80
        )

        plt.title("Association Rules: Support vs Confidence")
        plt.xlabel("Support")
        plt.ylabel("Confidence")

        plt.tight_layout()

        plt.savefig("graphs/support_vs_confidence.png")

        plt.close()

    # ==========================================================
    # 4. Top Rules by Lift
    # ==========================================================
    if not rules.empty:

        top_rules = rules.sort_values(
            "lift",
            ascending=False
        ).head(10).copy()

        top_rules["rule_name"] = (
            top_rules["antecedents"].apply(
                lambda x: ", ".join(x)
            )
            + " → "
            + top_rules["consequents"].apply(
                lambda x: ", ".join(x)
            )
        )

        plt.figure(figsize=(10, 6))

        plt.barh(
            top_rules["rule_name"],
            top_rules["lift"]
        )

        plt.title("Top 10 Association Rules by Lift")
        plt.xlabel("Lift")

        plt.tight_layout()

        plt.savefig("graphs/top_rules_by_lift.png")

        plt.close()

    # ==========================================================
    # 5. Confidence Distribution Histogram
    # ==========================================================
    if not rules.empty:

        plt.figure(figsize=(7, 4))

        plt.hist(
            rules["confidence"],
            bins=10
        )

        plt.title("Distribution of Rule Confidence")
        plt.xlabel("Confidence")
        plt.ylabel("Number of Rules")

        plt.tight_layout()

        plt.savefig("graphs/confidence_distribution.png")

        plt.close()

    # =========================
    # Return Results
    # =========================
    return {
        "apriori_time": round(apriori_time, 4),
        "fpgrowth_time": round(fpgrowth_time, 4),
        "rules": rules,
        "item_support": item_support
    }


if __name__ == "__main__":

    results = generate_and_train()

    print("Training complete.")

    print(results["rules"].head())