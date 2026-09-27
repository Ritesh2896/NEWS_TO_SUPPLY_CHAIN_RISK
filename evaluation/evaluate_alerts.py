import pandas as pd

def evaluate():
    df = pd.read_csv("data/processed/final_alerts.csv")
    if df.empty:
        print("No alerts.")
        return 0
    precision_proxy = df["risk_level"].isin(["HIGH","CRITICAL"]).mean()
    print("Alert Precision Proxy:", round(float(precision_proxy),3))
    return precision_proxy

if __name__ == "__main__":
    evaluate()
