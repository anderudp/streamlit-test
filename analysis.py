import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def invalids_pie_chart():
    df = pd.read_csv("diabetes.csv")
    df = df.drop(["Pregnancies", "DiabetesPedigreeFunction", "Age", "Outcome"], axis=1)
    df["invalids"] = df.isin([0, 0.0, np.nan]).sum(axis=1)
    counts = df.value_counts(["invalids"]).to_frame().reset_index().sort_values("invalids")

    #plt.bar_label("asd")
    fig, ax = plt.subplots()
    ax.pie(
        counts["count"], 
        labels=counts["invalids"],
        colors=["#54ff00", "#e4ff00", "#edaa00", "#f65500", "#870000"],
        explode=(0, 0, 0, 0.2, 0.2),
        autopct='%1.1f%%'
        )
    ax.set_title("Hiányzó adattagok száma a rekordokban")
    plt.savefig("counts.png")
    
    
def invalid_records_bar_chart():
    df = pd.read_csv("diabetes.csv")
    df = df.drop(["Pregnancies", "DiabetesPedigreeFunction", "Age", "Outcome"], axis=1)
    records = df.isin([0, 0.0, np.nan]).sum(axis=0).to_frame().reset_index().rename({0:"invalids"}, axis=1)
    records["valids"] = df.shape[0] - records["invalids"]
    
    plt.title("Érvényes adatok aránya")
    plt.bar(records["index"], records["invalids"], color="#cc0000")
    plt.bar(records["index"], records["valids"], bottom=records["invalids"], color="#009900")
    plt.savefig("bar.png")
    

def dpf_histogram():
    df = pd.read_csv("diabetes.csv")
    fig, ax = plt.subplots()
    plt.xlabel("DPF")
    plt.ylabel("Rekordok száma")
    ax.hist(
        df['DiabetesPedigreeFunction'],
        bins=100,
        histtype='stepfilled',
    )
    ax.set_title("DPF eloszlása az adathalmazban")
    plt.savefig("hist.png")
    
def insulin_histogram():
    df = pd.read_csv("diabetes.csv")
    df["invalids"] = df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].isin([0, 0.0, np.nan]).sum(axis=1)
    df.drop(df[df["invalids"] >= 3].index, inplace=True)
    
    df.drop(df[df["Insulin"] == 0.0].index, inplace=True)
    
    fig, ax = plt.subplots()
    plt.xlabel("Inzulin")
    plt.ylabel("Rekordok száma")
    ax.hist(
        df['Insulin'],
        bins=100,
        histtype='stepfilled',
    )
    ax.axvline(df['Insulin'].mean(), color='r', linestyle='dashed', linewidth=2)
    ax.set_title("Inzulin eloszlása az adathalmazban")
    plt.savefig("insulin_hist.png")
    
def features_histogram():
    df = pd.read_csv("diabetes.csv")
    df = df.iloc[:, :-1]
    for col in df.columns:
        fig, ax = plt.subplots()
        plt.xlabel(col)
        plt.ylabel("Rekordok száma")
        ax.hist(
            df[col],
            bins=80,
            histtype='stepfilled',
        )
        ax.axvline(df[col].mean(), color='r', linestyle='dashed', linewidth=2)
        ax.set_title(f"{col} oszlop eloszlása")
        plt.savefig(f"features/{col}_hist.png")
    
    
def skin_histogram():
    df = pd.read_csv("diabetes.csv")
    df["invalids"] = df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].isin([0, 0.0, np.nan]).sum(axis=1)
    df.drop(df[df["invalids"] >= 3].index, inplace=True)
    
    df.drop(df[df["SkinThickness"] == 0.0].index, inplace=True)
    
    fig, ax = plt.subplots()
    plt.xlabel("Bőrredővastagság")
    plt.ylabel("Rekordok száma")
    ax.hist(
        df['SkinThickness'],
        bins=80,
        histtype='stepfilled',
    )
    ax.axvline(df['SkinThickness'].mean(), color='r', linestyle='dashed', linewidth=2)
    ax.set_title("Bőrredővastagság eloszlása az adathalmazban")
    plt.savefig("skin_hist.png")
    
def outcome_bar():
    df = pd.read_csv("diabetes.csv")
    plt.title("Végkimenetek eloszlása")
    plt.bar(['Negatív', 'Pozitív'], df['Outcome'].value_counts().sort_index())
    plt.savefig('outcome_bar.png')
    
    print(df['Outcome'].value_counts().sort_index())
    
if __name__ == "__main__":
    outcome_bar()