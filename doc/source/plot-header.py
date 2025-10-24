import matplotlib.pyplot as plt

from heavyedge import ProfileData, get_sample_path
from heavyedge.api import mean_wasserstein

with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    Ys2, _, _ = data[:]
    mean2, L2 = mean_wasserstein(data, 100)

with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    Ys3, _, _ = data[:]
    mean3, L3 = mean_wasserstein(data, 100)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(Ys2.T, alpha=0.1, color="gray")
axes[0].plot(mean2[:L2])

axes[1].plot(Ys3.T, alpha=0.1, color="gray")
axes[1].plot(mean3[:L3])

axes[0].axis("off")
axes[1].axis("off")
fig.tight_layout()
