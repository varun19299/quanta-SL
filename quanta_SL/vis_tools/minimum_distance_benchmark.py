"""
Plotting minimum distance method from their output CSV
"""
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from quanta_SL.utils.plotting import save_plot

plt.style.use(["grid"])

params = {
    "legend.fontsize": 14,
    "figure.figsize": (7, 4),
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "font.family": "Times New Roman",
}
plt.rcParams.update(params)

csv_path = Path("outputs/benchmarks/KNN[Queries 102,400].csv", index_col=1)
df = pd.read_csv(csv_path)

df = df.drop(columns=["BCH [31, 11]"])
df["Method"] = df["Method"].str.replace("-", "\n")
df = df.set_index("Method")
df = df.iloc[::-1]

df.plot.barh(grid=True) #, color=["#36454F", "#808080", "#A9A9A9"][::-1])

plt.xlim(1e-3, 3e2)
plt.legend()
plt.tight_layout()
plt.xscale("log")
plt.ylabel(None)
plt.xlabel("Time (in seconds)")
# plt.title("Querying $512 x 256$ points")
save_plot(savefig=True, show=True, fname=csv_path.parent / f"{csv_path.stem}.pdf")
