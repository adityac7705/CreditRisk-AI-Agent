# The entire purpose of EDA is to ensure that training of our AI Model happens on correct data. It also helps us prove the usage of particular algorithms in used in our project. EDA makes sure that whatever work has been done so far is accurate and the collected data is now ready to be used for training.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "final_model_data.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "reports","figures")

def plot_eda() :
    if not os.path.exists(INPUT_PATH) :
        print(f"Error: File not found at {INPUT_PATH}. Please run preprocessing.py first.")
        return None
    
    os.makedirs(FIGURES_DIR, exist_ok = True)

    print("*---*Loading Data for Visualization*---*")
    df = pd.read_csv(INPUT_PATH)

    print("1. Plotting Target Distribution (Balance Check)...")
    plt.figure(figsize = (6, 4))
    sns.countplot(x = "target", data = df, palette = "coolwarm")
    plt.title("Loan Status Distribution (0 -> Paid, 1 -> Default)")
    plt.savefig(os.path.join(FIGURES_DIR, 'target_distribution.png'))
    plt.close()

    print("2. Plotting Correlation Heatmap...")
    plt.figure(figsize = (12, 10))
    corr = df.corr()
    top_cols = corr.index[abs(corr["target"]) > 0.05]

    sns.heatmap(df[top_cols].corr(), annot = True, cmap = "RdBu", fmt = ".2f")
    plt.title("Correlation Matrix (Features vs Target)")
    plt.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"))
    plt.close()

    print("3. Plotting Income vs Default Risk...")
    plt.figure(figsize = (8, 6))

    sns.boxplot(x = "target", y = "annual_inc", data = df, palette = "Set2")
    plt.ylim(0, 150000)
    plt.title("Annual Income by Loan Status")
    plt.savefig(os.path.join(FIGURES_DIR, "income_boxplot.png"))
    plt.close()

    print(f"EDA Complete! Check the folder: {FIGURES_DIR}")

if __name__ == "__main__" :
    plot_eda()
