from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

albedo_ll = [0.1, 0.3, 0.5, 1.0]
df_dict = {
    albedo: pd.read_csv(f"outputs/sphere/diffuse_{albedo}/ConventionalGray/mean_l1.csv")
    for albedo in albedo_ll
}

exposure_ll = df_dict[0.1]["Exposure"]
intensity_multiplier_ll = df_dict[0.1]["Intensity Multiplier"]
color_ll = ["r", "b", "g", "orange"]

for albedo, color in zip(albedo_ll, color_ll):
    plt.semilogx(
        exposure_ll, df_dict[albedo]["Mean L1"], color=color, marker="o", linewidth=2
    )

legend = [f"albedo={albedo}" for albedo in albedo_ll]
plt.legend(legend)
plt.xlabel("$t_\mathrm{exp}$")
plt.ylabel("Mean L1 Error")
plt.title("Mean L1 error vs Exposure")
plt.grid()
plt.savefig("outputs/plots/albedo_vs_l1.pdf", dpi=150)
plt.show()
