import torch
import numpy as np
import matplotlib.pyplot as plt


def flatten_list_of_arrays(lst):
    return np.concatenate([np.asarray(x).ravel() for x in lst])

FS_TITLE = 13
FS_LABEL = 13
FS_TICK = 13
FS_SUPTITLE = 14
FS_ANNOT = 13

# Load pickle files
fm = torch.load("test_values_fm.pkl")
mh = torch.load("test_values_mh.pkl")

# Plot configuration
keys = [
    ("energy", "exact_energy", "Energy", False),
    ("force_x", "exact_force_x", "Force (X)", True),
    ("force_y", "exact_force_y", "Force (Y)", True),
    ("force_z", "exact_force_z", "Force (Z)", True),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
axes = axes.flatten()

for ax, (pred_key, exact_key, title, is_force) in zip(axes, keys):

    if is_force:
        exact = flatten_list_of_arrays(fm[exact_key])
        pred_fm = flatten_list_of_arrays(fm[pred_key])
        pred_mh = flatten_list_of_arrays(mh[pred_key])
    else:
        exact = np.asarray(fm[exact_key])
        pred_fm = np.asarray(fm[pred_key])
        pred_mh = np.asarray(mh[pred_key])

    ax_r = ax.twinx()

    # FM
    ax.scatter(
        exact, 
        pred_fm, 
        s=13, 
        alpha=0.6, 
        color="royalblue", 
    )

    # MH
    ax_r.scatter(
        exact,
        pred_mh,
        s=13,
        alpha=0.6,
        color="crimson",
        marker="o",
    )

    # y = x
    min_val = min(exact.min(), pred_fm.min(), pred_mh.min())
    max_val = max(exact.max(), pred_fm.max(), pred_mh.max())

    ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1, label=r"$y = x$")

    ax.set_ylim(min_val, max_val)
    ax_r.set_ylim(min_val, max_val)

    ax.set_title(title, fontsize=FS_TITLE)
    ax.set_xlabel("Exact", fontsize=FS_LABEL)
    ax.set_ylabel("Foundation Model Prediction", fontsize=FS_LABEL, color="royalblue")
    ax_r.set_ylabel("Fine-Tuned Model Prediction", fontsize=FS_LABEL, color="crimson")

    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.tick_params(axis="y", labelcolor="royalblue")
    ax_r.tick_params(axis="y", labelsize=FS_TICK, labelcolor="crimson")

    ax.grid(False)
    ax.legend(fontsize=FS_ANNOT)

# Suptitle
plt.suptitle(
    "Exact vs Prediction (FM vs MH)",
    fontsize=FS_SUPTITLE
)

plt.suptitle("Exact vs Prediction (FM vs MH)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("exact_vs_predict_fm_mh_2x2.png", dpi=300)
plt.show()
