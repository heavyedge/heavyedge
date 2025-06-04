import matplotlib.pyplot as plt

from heavyedge import ProfileData, get_sample_path
from heavyedge.api import (
    landmarks_type2,
    landmarks_type3,
    mean,
    plateau_type2,
    plateau_type3,
)

with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    x2 = data.x()
    profiles2 = list(data.profiles())
mean2 = mean(profiles2, 1000)
lm2 = landmarks_type2(mean2, 32)
b0_2, b1_2, psi_2 = plateau_type2(x2[: len(mean2)], mean2, lm2[1], lm2[2])

with ProfileData(get_sample_path("Prep-Type3.h5")) as data:
    x3 = data.x()
    profiles3 = list(data.profiles())
mean3 = mean(profiles3, 1000)
lm3 = landmarks_type3(mean3, 32)
b0_3, b1_3, psi_3 = plateau_type3(x3[: len(mean3)], mean3, lm3[2], lm3[3])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for profile in profiles2:
    axes[0].plot(x2[: len(profile)], profile, alpha=0.1, color="gray")
axes[0].plot(x2[: len(mean2)], mean2, color="gray")
axes[0].plot(x2[lm2[:-1]], mean2[lm2[:-1]], "o")
X2 = x2[x2 < psi_2]
axes[0].plot(X2, b0_2 + b1_2 * X2)

for profile in profiles3:
    axes[1].plot(x3[: len(profile)], profile, alpha=0.1, color="gray")
axes[1].plot(x3[: len(mean3)], mean3, color="gray")
axes[1].plot(x3[lm3[:-1]], mean3[lm3[:-1]], "o")
X3 = x3[x3 < psi_3]
axes[1].plot(X3, b0_3 + b1_3 * X3)

axes[0].axis("off")
axes[1].axis("off")
fig.tight_layout()
