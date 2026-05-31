import gvar as gv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


Ls = 32

n_values = np.array([0, 1, 2, 3, 4, 5], dtype=float)

E_values = gv.gvar([
    "0.14169(69)", # nstate=4 t=4-14
    "0.24422(67)", # nstate=2 t=4-14
    "0.4183(20)",  # nstate=2 t=4-14
    "0.6068(56)",  # nstate=2 t=4-14
    "0.8005(95)",  # nstate=2 t=4-14
    "0.979(13)",   # nstate=2 t=4-14
])

p_values = 2.0 * np.pi * n_values / Ls
p2_values = p_values**2
E2_values = E_values**2

m0 = E_values[0]
m02 = m0**2

p2_curve = np.linspace(0.0, max(p2_values) * 1.05, 300)
E2_curve = m02 + p2_curve

plt.figure(figsize=(7, 5))

plt.errorbar(
    p2_values,
    gv.mean(E2_values),
    yerr=gv.sdev(E2_values),
    fmt="o",
    mfc="white",
    capsize=4,
    label="data: E^2",
)

plt.plot(
    p2_curve,
    gv.mean(E2_curve),
    "--",
    color="tab:orange",
    label="E^2 = E0^2 + p^2",
)

plt.fill_between(
    p2_curve,
    gv.mean(E2_curve) - gv.sdev(E2_curve),
    gv.mean(E2_curve) + gv.sdev(E2_curve),
    color="tab:orange",
    alpha=0.25,
    label="E0 error band",
)

for n, x, y in zip(n_values, p2_values, gv.mean(E2_values)):
    plt.annotate(
        f"p{int(n)}",
        (x, y),
        textcoords="offset points",
        xytext=(5, 5),
    )

plt.xlabel("(p a)^2 = (2 pi n / Ls)^2")
plt.ylabel("(E a)^2")
plt.title("Relativistic dispersion relation")
plt.grid(linestyle=":")
plt.legend()
plt.tight_layout()

output_path = "dispersion_E2_vs_p2.png"
plt.savefig(output_path, dpi=300)

print("saved plot to", output_path)
