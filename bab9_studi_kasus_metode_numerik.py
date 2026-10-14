import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path
 
# Parameter dan dataset kanonis
T_amb = 25.0
T0 = 80.0
k = 0.15
q = 1.5
t = np.arange(0, 21, 1, dtype=float)
noise = np.array([
    0.0, 0.8, -0.5, 0.6, -0.7, 0.4, -0.3,
    0.5, -0.6, 0.2, -0.4, 0.3, -0.2, 0.4,
    -0.3, 0.2, -0.2, 0.1, -0.1, 0.1, -0.05
], dtype=float)
 
def exact_solution(time, q_value=q):
    """Solusi analitik model dT/dt = -k(T - T_amb) + q."""
    time = np.asarray(time, dtype=float)
    return (T_amb + q_value / k
            + (T0 - T_amb - q_value / k) * np.exp(-k * time))
 
def thermal_model(time, temperature, q_value=q):
    """Sisi kanan persamaan diferensial termal."""
    return -k * (temperature - T_amb) + q_value
 
T_true = exact_solution(t)
T_obs = T_true + noise
dt = float(t[1] - t[0])
 
data = pd.DataFrame({
    "t": t,
    "T_true": T_true,
    "noise": noise,
    "T_obs": T_obs,
})
 
data["dTdt_central"] = np.nan
data.loc[1:len(data)-2, "dTdt_central"] = (
    data["T_obs"].to_numpy()[2:] - data["T_obs"].to_numpy()[:-2]
) / (2.0 * dt)
 
integral_obs = float(np.trapezoid(T_obs, t))
integral_true = float(np.trapezoid(T_true, t))
 
def solve_euler(time, q_value=q):
    """Euler eksplisit untuk grid waktu yang diberikan."""
    time = np.asarray(time, dtype=float)
    values = np.empty(time.size, dtype=float)
    values[0] = T0
    for i in range(time.size - 1):
        h = time[i + 1] - time[i]
        values[i + 1] = values[i] + h * thermal_model(
            time[i], values[i], q_value
        )
    return values
 
def solve_rk4(time, q_value=q):
    """Runge-Kutta orde empat klasik pada grid waktu yang diberikan."""
    time = np.asarray(time, dtype=float)
    values = np.empty(time.size, dtype=float)
    values[0] = T0
    for i in range(time.size - 1):
        h = time[i + 1] - time[i]
        ti = time[i]
        yi = values[i]
        k1 = thermal_model(ti, yi, q_value)
        k2 = thermal_model(ti + h / 2.0, yi + h * k1 / 2.0, q_value)
        k3 = thermal_model(ti + h / 2.0, yi + h * k2 / 2.0, q_value)
        k4 = thermal_model(ti + h, yi + h * k3, q_value)
        values[i + 1] = yi + h * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return values
 
T_euler = solve_euler(t)
T_rk4 = solve_rk4(t)
error_euler = T_euler - T_true
error_rk4 = T_rk4 - T_true
residual_obs_rk4 = T_obs - T_rk4
rmse_euler = float(np.sqrt(np.mean(error_euler ** 2)))
rmse_rk4 = float(np.sqrt(np.mean(error_rk4 ** 2)))
rmse_obs_rk4 = float(np.sqrt(np.mean(residual_obs_rk4 ** 2)))
 
data["T_euler"] = T_euler
data["T_rk4"] = T_rk4
data["error_euler"] = error_euler
data["error_rk4"] = error_rk4
data["residual_obs_rk4"] = residual_obs_rk4
 
t_interp = 14.5
T_interp_true = float(np.interp(t_interp, t, T_true))
T_interp_rk4 = float(np.interp(t_interp, t, T_rk4))
T_interp_obs = float(np.interp(t_interp, t, T_obs))
 
def bisection(func, a, b, eps_x=1e-8, eps_f=1e-8, n_max=100):
    """Bisection dengan syarat delta dan residual sekaligus."""
    a = float(a)
    b = float(b)
    fa = float(func(a))
    fb = float(func(b))
    if not (np.isfinite(fa) and np.isfinite(fb)):
        raise ValueError("Nilai fungsi pada ujung interval harus finite.")
    if fa == 0.0:
        return {"root": a, "residual": 0.0, "delta": None,
                "iterations": 0, "converged": True, "bracket": (a, a),
                "history": []}
    if fb == 0.0:
        return {"root": b, "residual": 0.0, "delta": None,
                "iterations": 0, "converged": True, "bracket": (b, b),
                "history": []}
    if fa * fb > 0.0:
        raise ValueError("Bisection memerlukan perubahan tanda pada interval.")
    history = []
    x_prev = None
    x_mid = None
    residual = None
    delta = None
    converged = False
    for iteration in range(1, n_max + 1):
        x_mid = (a + b) / 2.0
        f_mid = float(func(x_mid))
        residual = abs(f_mid)
        delta = None if x_prev is None else abs(x_mid - x_prev)
        history.append({
            "iteration": iteration, "x_n": x_mid,
            "residual": residual, "delta": delta,
            "a": a, "b": b
        })
        if delta is not None and delta <= eps_x and residual <= eps_f:
            converged = True
            break
        if f_mid == 0.0:
            # Akar eksak adalah kasus khusus: residual nol menjamin akar ditemukan.

            converged = True
            break
        if fa * f_mid < 0.0:
            b, fb = x_mid, f_mid
        else:
            a, fa = x_mid, f_mid
        x_prev = x_mid
    return {"root": float(x_mid), "residual": float(residual),
            "delta": None if delta is None else float(delta),
            "iterations": len(history), "converged": converged,
            "bracket": (a, b), "history": history}
 
def root_function(time):
    """Residual suhu dari interpolasi linear kurva RK4 terhadap target 40."""
    return float(np.interp(float(time), t, T_rk4) - 40.0)
 
root_result = bisection(root_function, 14.0, 15.0)
analytic_root = float(-np.log((40.0 - T_amb - q/k) /
                              (T0 - T_amb - q/k)) / k)
 
def objective(q_value, target=40.0):
    """Kuadrat selisih suhu akhir RK4 terhadap target."""
    final_temperature = float(solve_rk4(t, q_value)[-1])
    return (final_temperature - target) ** 2
 
def golden_section_search(func, a, b, eps_x=1e-4, n_max=100):
    """Minimisasi Golden Section Search dengan evaluasi fungsi yang digunakan ulang."""
    a = float(a)
    b = float(b)
    if not (a < b):
        raise ValueError("Batas harus memenuhi a < b.")
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = b - ratio * (b - a)
    x2 = a + ratio * (b - a)
    f1 = float(func(x1))
    f2 = float(func(x2))
    history = []
    converged = False
    for iteration in range(1, n_max + 1):
        width = b - a
        history.append({"iteration": iteration, "a": a, "b": b,
                        "width": width, "x1": x1, "x2": x2,
                        "f1": f1, "f2": f2})
        if width <= eps_x:
            converged = True
            break
        if f1 <= f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - ratio * (b - a)
            f1 = float(func(x1))
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + ratio * (b - a)
            f2 = float(func(x2))
    # Periksa lebar interval setelah pembaruan terakhir, termasuk saat batas iterasi tercapai.
    if b - a <= eps_x:
        converged = True
    x_opt = (a + b) / 2.0
    return {"x_opt": float(x_opt), "f_opt": float(func(x_opt)),
            "interval": (a, b), "width": float(b-a),
            "iterations": len(history), "converged": converged,
            "history": history}
 
golden_result = golden_section_search(objective, 0.0, 4.0, eps_x=1e-4, n_max=100)
analytic_q_opt = float(
    k * (40.0 - T_amb - (T0 - T_amb) * np.exp(-k * t[-1]))
    / (1.0 - np.exp(-k * t[-1]))
)
 
# Grafik disimpan sebagai artefak dan juga dirender ke buffer untuk pemeriksaan otomatis.
figure_dir = Path(__file__).with_name("Bab9_Revision8_Grafik")
figure_dir.mkdir(parents=True, exist_ok=True)
figure_buffers = {}
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(t, T_obs, "o", label="T_obs")
ax.plot(t, T_true, "-", label="T_true (analitik)")
ax.plot(t, T_euler, "--", label="Euler")
ax.plot(t, T_rk4, ":", label="RK4")
ax.set(xlabel="Waktu", ylabel="Suhu", title="Data dan model termal")
ax.grid(True)
ax.legend()
fig.savefig(figure_dir / "Gambar_9_1_Data_dan_Model.png", format="png", dpi=160, bbox_inches="tight")
buf = BytesIO()
fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
figure_buffers["data_model"] = buf.getvalue()
plt.close(fig)
 
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.axhline(0.0, linewidth=1)
ax.plot(t, residual_obs_rk4, "o-")
ax.set(xlabel="Waktu", ylabel="Residual T_obs - T_RK4",
       title="Residual observasi terhadap RK4")
ax.grid(True)
fig.savefig(figure_dir / "Gambar_9_2_Residual_RK4.png", format="png", dpi=160, bbox_inches="tight")
buf = BytesIO()
fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
figure_buffers["residual"] = buf.getvalue()
plt.close(fig)
 
q_grid = np.linspace(0.0, 4.0, 201)
J_grid = np.array([objective(qi) for qi in q_grid])
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(q_grid, J_grid, label="J(q)")
ax.axvline(golden_result["x_opt"], linestyle="--", label="q optimum")
ax.set(xlabel="q", ylabel="J(q)", title="Fungsi objektif optimasi")
ax.grid(True)
ax.legend()
# Inset memperbesar daerah minimum tanpa mengganti grafik dasar.
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins = inset_axes(ax, width="43%", height="43%", loc="lower right", borderpad=1.2)
q_local = np.linspace(golden_result["x_opt"] - 0.01, golden_result["x_opt"] + 0.01, 301)
J_local = np.array([objective(qi) for qi in q_local])
axins.plot(q_local, J_local)
axins.axvline(golden_result["x_opt"], linestyle="--", linewidth=1)
axins.set_xlim(float(q_local[0]), float(q_local[-1]))
axins.set_xticks(np.linspace(float(q_local[0]), float(q_local[-1]), 3))
axins.set_xticklabels([f"{value:.3f}" for value in np.linspace(float(q_local[0]), float(q_local[-1]), 3)])
axins.set_ylim(0.0, max(float(np.max(J_local)) * 1.05, 1e-12))
axins.set_title("Pembesaran daerah minimum", fontsize=8)
axins.tick_params(labelsize=7)
axins.grid(True)
fig.savefig(figure_dir / "Gambar_9_3_Fungsi_Objektif.png", format="png", dpi=160, bbox_inches="tight")
buf = BytesIO()
fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
figure_buffers["objective"] = buf.getvalue()
plt.close(fig)
 
stats = {
    "n": int(len(T_obs)),
    "min": float(np.min(T_obs)),
    "max": float(np.max(T_obs)),
    "mean": float(np.mean(T_obs)),
    "median": float(np.median(T_obs)),
    "sample_std": float(np.std(T_obs, ddof=1)),
}
 
# Pemeriksaan otomatis yang menghitung hasil dari variabel yang baru dibuat.
checks = {
    "panjang_data_21": len(t) == 21 and len(noise) == 21,
    "nilai_numerik_finite": bool(np.isfinite(
        np.concatenate([T_true, T_obs, T_euler, T_rk4, error_euler,
                        error_rk4, residual_obs_rk4])).all()),
    "keluaran_euler_rk4": T_euler.shape == t.shape and T_rk4.shape == t.shape,
    "statistik_finite": bool(np.isfinite(list(stats.values())).all()),
    "statistik_sesuai_acuan": (
        np.isclose(stats["min"], 37.190418076554, atol=5e-12, rtol=0.0)
        and np.isclose(stats["max"], 80.0, atol=5e-12, rtol=0.0)
        and np.isclose(stats["mean"], 49.736589665416, atol=5e-12, rtol=0.0)
        and np.isclose(stats["median"], 44.640857206679, atol=5e-12, rtol=0.0)
        and np.isclose(stats["sample_std"], 12.800010417193, atol=5e-12, rtol=0.0)
    ),
    "integral_finite": np.isfinite(integral_obs) and np.isfinite(integral_true),
    "integral_sesuai_acuan": (
        np.isclose(integral_obs, 985.873173935460, atol=5e-12, rtol=0.0)
        and np.isclose(integral_true, 985.598173935459, atol=5e-12, rtol=0.0)
    ),
    "rmse_finite": np.isfinite(rmse_euler) and np.isfinite(rmse_rk4) and np.isfinite(rmse_obs_rk4),
    "rmse_sesuai_acuan": (
        np.isclose(rmse_euler, 0.972556131174, atol=5e-12, rtol=0.0)
        and np.isclose(rmse_rk4, 0.000058909958, atol=5e-12, rtol=0.0)
        and np.isclose(rmse_obs_rk4, 0.395962984826, atol=5e-12, rtol=0.0)
    ),
    "interval_bisection_valid": (
        root_result["bracket"][0] <= root_result["bracket"][1]
        and root_function(root_result["bracket"][0])
            * root_function(root_result["bracket"][1]) <= 0.0
    ),
    "bisection_delta_tolerance": root_result["delta"] is not None and root_result["delta"] <= 1e-8,
    "bisection_residual_tolerance": root_result["residual"] <= 1e-8,
    "bisection_converged": root_result["converged"],
    "bisection_matches_reference": (
        np.isclose(root_result["root"], 14.665201835334, atol=5e-12, rtol=0.0)
        and root_result["iterations"] == 27
    ),
    "golden_interval_tolerance": golden_result["width"] <= 1e-4,
    "golden_converged": golden_result["converged"],
    "golden_matches_reference": (
        np.isclose(golden_result["x_opt"], 1.935635409319, atol=5e-12, rtol=0.0)
        and np.isclose(golden_result["f_opt"], 8.245227833015e-9, atol=5e-20, rtol=0.0)
        and golden_result["iterations"] == 24
    ),
    "optimum_inside_bounds": 0.0 <= golden_result["x_opt"] <= 4.0,
    "three_graphs_rendered": len(figure_buffers) == 3 and all(len(v) > 0 for v in figure_buffers.values()),
}
 
# Kasus tepi: akar tepat di titik tengah pada iterasi pertama.
edge_mid = bisection(lambda x: x - 0.5, 0.0, 1.0, eps_x=1e-8, eps_f=1e-8)
edge_midpoint_ok = (edge_mid["root"] == 0.5 and edge_mid["iterations"] == 1
                    and edge_mid["residual"] == 0.0 and edge_mid["converged"])
 
# Kasus tepi tambahan untuk memastikan validasi interval dan akar ujung.
edge_endpoint = bisection(lambda x: x - 2.0, 2.0, 5.0)
endpoint_ok = edge_endpoint["root"] == 2.0 and edge_endpoint["residual"] == 0.0
invalid_bracket_rejected = False
try:
    bisection(lambda x: x*x + 1.0, -1.0, 1.0)
except ValueError:
    invalid_bracket_rejected = True
 
checks.update({
    "bisection_exact_endpoint_handled": endpoint_ok,
    "bisection_exact_midpoint_handled": edge_midpoint_ok,
    "bisection_invalid_bracket_rejected": invalid_bracket_rejected,
    "bisection_root_matches_interpolated_rk4": abs(root_function(root_result["root"])) <= 1e-8,
    "golden_bounds_canonical_0_4": 0.0 <= golden_result["x_opt"] <= 4.0,
})
 
# Ringkasan aktual yang digunakan untuk audit.
print("ENVIRONMENT", f"Python={__import__('sys').version.split()[0]}", f"NumPy={np.__version__}", f"Pandas={pd.__version__}", f"Matplotlib={matplotlib.__version__}")
print("FIGURE_DIR", str(figure_dir))
print("STATS", stats)
print("INTEGRALS", f"{integral_obs:.15f}", f"{integral_true:.15f}")
print("RMSE", f"{rmse_euler:.15f}", f"{rmse_rk4:.15f}", f"{rmse_obs_rk4:.15f}")
print("BISECTION", f"f14={root_function(14):.15f}", f"f15={root_function(15):.15f}",
      f"root={root_result['root']:.15f}", f"residual={root_result['residual']:.15e}",
      f"delta={root_result['delta']:.15e}", f"iterations={root_result['iterations']}",
      f"converged={root_result['converged']}", f"analytic={analytic_root:.15f}")
T_final_opt = float(solve_rk4(t, golden_result["x_opt"])[-1])
print("GOLDEN", f"q_opt={golden_result['x_opt']:.15f}",
      f"J_opt={golden_result['f_opt']:.15e}",
      f"T_final={T_final_opt:.15f}",
      f"interval=({golden_result['interval'][0]:.15f}, {golden_result['interval'][1]:.15f})",
      f"width={golden_result['width']:.15e}",
      f"iterations={golden_result['iterations']}", f"converged={golden_result['converged']}",
      f"analytic_q={analytic_q_opt:.15f}")
print("INTERPOLATION", f"T_true={T_interp_true:.12f}",
      f"T_rk4={T_interp_rk4:.12f}", f"T_obs={T_interp_obs:.12f}")
print("FIGURES", {name: len(content) for name, content in figure_buffers.items()})
print("CHECKS", checks)
print("EDGE_TESTS", {
    "endpoint_root_handled": endpoint_ok,
    "midpoint_first_iteration_exact_root_converged": edge_midpoint_ok,
    "invalid_bracket_rejected": invalid_bracket_rejected,
})
