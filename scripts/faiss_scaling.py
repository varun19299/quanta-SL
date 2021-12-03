"""
Plotting FAISS scaling from their output CSV
"""
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from quanta_SL.utils.plotting import save_plot

plt.style.use(["grid","high-contrast"])

params = {
    "legend.fontsize": 14,
    "figure.figsize": (6, 4),
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "font.family": "Times New Roman",
}
plt.rcParams.update(params)

csv_path = Path("outputs/benchmarks/FAISS [Queries 102,400] scaling.csv")
df = pd.read_csv(csv_path)

df = df.drop(columns=["BCH [31, 11]"])

device_ll = ["CPU", "GPU"]
ylim_ll = [(5e-2, 4e1),(1e-3, 1e-1)]

for device, ylim in zip(device_ll, ylim_ll):
    sub_df = df[df["Device"] == device]
    sub_df = sub_df.set_index("Size")
    sub_df = sub_df.iloc[::-1]

    sub_df.plot.bar(grid=True, rot=0)  # , color=["#36454F", "#808080", "#A9A9A9"][::-1])

    plt.ylim(ylim)
    plt.legend()
    plt.tight_layout()
    plt.yscale("log")
    plt.ylabel("Time (in seconds)")

    plt.xlabel(None)
    # plt.title("Querying $512 x 256$ points")
    save_plot(savefig=True, show=True, fname=csv_path.parent / f"{csv_path.stem}_device_{device}.pdf")
