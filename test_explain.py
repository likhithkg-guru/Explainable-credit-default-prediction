import pandas as pd
from explainability import (Explainer, plot_contributions,
                            plot_global_importance, DISCLAIMER)

X_test = pd.read_csv("data/X_test.csv")

for model_name in ["decision_tree", "random_forest", "xgboost"]:
    print(f"\n=========== {model_name} ===========")
    ex = Explainer(model_name)

    # Global importance
    g = ex.shap_global(X_test)
    print("Top 5 features overall:\n", g.head(5).to_string(index=False))
    plot_global_importance(g, f"{model_name}: overall importance").savefig(
        f"assets/shap_global_{model_name}.png", dpi=150)

    # Two example customers: highest-risk and lowest-risk in the test set
    probs = ex.model.predict_proba(X_test)[:, 1]
    for tag, idx in [("high_risk", probs.argmax()), ("low_risk", probs.argmin())]:
        row = X_test.iloc[[idx]]
        print(f"\n{tag} customer #{idx}: estimated default probability = {probs[idx]:.2f}")

        s = ex.shap_local(row)
        print("SHAP top 5:\n", s[["label", "shap_value"]].head(5).to_string(index=False))
        plot_contributions(s, "label", "shap_value",
                           f"SHAP: {model_name} ({tag})",
                           f"SHAP value ({ex.units})").savefig(
            f"assets/shap_{model_name}_{tag}.png", dpi=150)

        l = ex.lime_local(row)
        print("LIME top 5:\n", l.head(5).to_string(index=False))
        plot_contributions(l, "rule", "weight",
                           f"LIME: {model_name} ({tag})",
                           "LIME weight").savefig(
            f"assets/lime_{model_name}_{tag}.png", dpi=150)

print("\n" + DISCLAIMER)