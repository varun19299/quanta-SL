from matplotlib import pyplot as plt
from quanta_SL.utils.plotting import arrowed_spines

# Matplotlib font sizes
TINY_SIZE = 8
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)

# removing the default axis on all sides:
for side in ["right", "top", "bottom", "left"]:
    ax.spines[side].set_visible(False)
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_yticks([0, 1, 2, 3, 4])

ax.set_xticklabels(["1Hz", "10Hz", "100Hz", "1000Hz", "10kHz"])
# ax.set_xlabel("Speed")


ax.set_yticklabels(["100cm", "10cm", "1cm", "1mm", "$100\mu m$"])
# ax.set_ylabel("Depth Resolution")
annots = arrowed_spines(ax, locations=("bottom right", "left up"))

plt.tight_layout()
plt.grid()

plt.savefig("outputs/plots/broader_impact.pdf", dpi=150)
plt.show()
