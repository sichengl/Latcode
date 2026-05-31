import os

import gvar as gv
import lsqfit as lsf
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


input_path = "./gvar_data/pdf_O3_conn_gvar_jk_manual.p"
psink_key = "p0"
pt2_real_sign = -1.0
tmin = 3
tmax = 14
nstate = 4
output_path = f"./plots/pt2_fit_{psink_key}_nstate{nstate}.png"
zoom_output_path = f"./plots/pt2_fit_{psink_key}_nstate{nstate}_zoom.png"
plot_xmin = tmin - 0.5
plot_xmax = tmax - 0.5
plot_y_pad_fraction = 0.15

prior = gv.BufferDict()
prior["E0"] = gv.gvar(0.5, 0.5)
prior["z0"] = gv.gvar(1.0, 10.0)

for state in range(1, nstate):
    prior[f"log(dE{state})"] = gv.gvar(1.0, 5.0)
    prior[f"z{state}"] = gv.gvar(0.0, 10.0)


data = gv.load(input_path)

if "Lt" in data:
    Lt = int(data["Lt"])
elif "Gt" in data:
    Lt = int(data["Gt"])
else:
    Lt = 96

t_all = np.array(list(data["pt2_tsep_list"]), dtype=float)

c2_raw = list(data["pt2_by_psink"][psink_key]["real"])
c2_all = [pt2_real_sign * y for y in c2_raw]

t_fit = []
c2_fit = []

for t, y in zip(t_all, c2_all):
    if tmin <= t < tmax:
        t_fit.append(t)
        c2_fit.append(y)

t_fit = np.array(t_fit, dtype=float)

def pt2_re_fcn(pt2_t, params, Lt, nstate=2):
    val = 0.0
    energy = params["E0"]

    for state in range(nstate):
        if state > 0:
            energy = energy + params[f"dE{state}"]

        z = params[f"z{state}"]

        val = val + z**2 / (2.0 * energy) * (
            np.exp(-energy * pt2_t) + np.exp(-energy * (Lt - pt2_t))
        )

    return val


def fit_function(t, params):
    return pt2_re_fcn(t, params, Lt, nstate)


fit = lsf.nonlinear_fit(
    data=(t_fit, c2_fit),
    prior=prior,
    fcn=fit_function,
    maxit=10000,
)

print(fit)

print("E0 =", fit.p["E0"])

for state in range(1, nstate):
    print(f"dE{state} =", fit.p[f"dE{state}"])

for state in range(nstate):
    print(f"z{state} =", fit.p[f"z{state}"])

print("Q =", fit.Q)
print("chi2/dof =", fit.chi2 / fit.dof)

t_plot = np.linspace(min(t_all), max(t_all), 300)
c2_plot = pt2_re_fcn(t_plot, fit.p, Lt, nstate)

c2_plot_mean = gv.mean(c2_plot)
c2_plot_sdev = gv.sdev(c2_plot)

c2_fit_model = pt2_re_fcn(t_fit, fit.p, Lt, nstate)

plt.figure(figsize=(7, 5))

plt.axvspan(
    tmin,
    tmax - 1,
    color="gray",
    alpha=0.2,
    label="fit region",
)

plt.errorbar(
    t_all,
    gv.mean(c2_all),
    yerr=gv.sdev(c2_all),
    fmt="o",
    mfc="white",
    capsize=4,
    label="data",
)

plt.errorbar(
    t_fit,
    gv.mean(c2_fit_model),
    yerr=gv.sdev(c2_fit_model),
    fmt="s",
    capsize=4,
    label="fit model at fit points",
)

plt.plot(
    t_plot,
    c2_plot_mean,
    color="tab:orange",
    label="fit curve",
)

plt.fill_between(
    t_plot,
    c2_plot_mean - c2_plot_sdev,
    c2_plot_mean + c2_plot_sdev,
    color="tab:orange",
    alpha=0.3,
    label="fit error band",
)

plt.xlabel("t_sep / a")
plt.ylabel("-C2 real")
plt.title(
    f"{psink_key}, nstate={nstate}, "
    f"chi2/dof={fit.chi2 / fit.dof:.3g}, Q={fit.Q:.3g}"
)
plt.grid(linestyle=":")
plt.legend()
plt.tight_layout()

output_dir = os.path.dirname(output_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

plt.savefig(output_path, dpi=300)

print("saved plot to", output_path)

zoom_mask_data = (t_all >= plot_xmin) & (t_all <= plot_xmax)
zoom_mask_curve = (t_plot >= plot_xmin) & (t_plot <= plot_xmax)

zoom_y_low = []
zoom_y_high = []

for y in np.array(c2_all, dtype=object)[zoom_mask_data]:
    zoom_y_low.append(gv.mean(y) - gv.sdev(y))
    zoom_y_high.append(gv.mean(y) + gv.sdev(y))

for y_mean, y_sdev in zip(c2_plot_mean[zoom_mask_curve], c2_plot_sdev[zoom_mask_curve]):
    zoom_y_low.append(y_mean - y_sdev)
    zoom_y_high.append(y_mean + y_sdev)

zoom_ymin = min(zoom_y_low)
zoom_ymax = max(zoom_y_high)
zoom_ypad = plot_y_pad_fraction * (zoom_ymax - zoom_ymin)

plt.xlim(plot_xmin, plot_xmax)
plt.ylim(zoom_ymin - zoom_ypad, zoom_ymax + zoom_ypad)

plt.savefig(zoom_output_path, dpi=300)

print("saved zoom plot to", zoom_output_path)
