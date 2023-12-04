import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def invalids_pie_chart():
    df = pd.read_csv("diabetes.csv")
    df = df.drop(["Pregnancies", "DiabetesPedigreeFunction", "Age", "Outcome"], axis=1)
    df["invalids"] = df.isin([0, 0.0, np.nan]).sum(axis=1)
    counts = df.value_counts(["invalids"]).to_frame().reset_index().sort_values("invalids")

    fig, ax = plt.subplots()
    ax.pie(
        counts["count"], 
        labels=counts["invalids"],
        colors=["#54ff00", "#e4ff00", "#edaa00", "#f65500", "#870000"],
        explode=(0, 0, 0, 0.2, 0.2)
        )

    plt.savefig("counts.png")
    
    
def invalid_records_bar_chart():
    df = pd.read_csv("diabetes.csv")
    df = df.drop(["Pregnancies", "DiabetesPedigreeFunction", "Age", "Outcome"], axis=1)
    records = df.isin([0, 0.0, np.nan]).sum(axis=0).to_frame().reset_index().rename({0:"invalids"}, axis=1)
    records["valids"] = df.shape[0] - records["invalids"]
    
    plt.bar(records["index"], records["invalids"], color="#cc0000")
    plt.bar(records["index"], records["valids"], bottom=records["invalids"], color="#009900")
    plt.savefig("bar.png")
    
    
if __name__ == "__main__":
    invalid_records_bar_chart()