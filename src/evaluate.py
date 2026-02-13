"""
Оценка моделей: метрики, сравнение, симуляция урожайности.

Запуск:
    python evaluate.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ========================== СИМУЛЯЦИЯ УРОЖАЯ ==========================

CROP_PARAMS = {
    "wheat":     {"yield_max": 6.0,  "c_N": 0.020, "c_P": 0.035, "c_K": 0.040,
                  "temp_opt": 18, "ph_opt": 6.5},
    "rice":      {"yield_max": 7.5,  "c_N": 0.022, "c_P": 0.030, "c_K": 0.035,
                  "temp_opt": 28, "ph_opt": 6.0},
    "maize":     {"yield_max": 9.0,  "c_N": 0.015, "c_P": 0.025, "c_K": 0.030,
                  "temp_opt": 25, "ph_opt": 6.2},
    "soybean":   {"yield_max": 3.5,  "c_N": 0.050, "c_P": 0.030, "c_K": 0.025,
                  "temp_opt": 26, "ph_opt": 6.3},
    "potato":    {"yield_max": 35.0, "c_N": 0.012, "c_P": 0.020, "c_K": 0.015,
                  "temp_opt": 18, "ph_opt": 5.8},
    "cotton":    {"yield_max": 4.0,  "c_N": 0.018, "c_P": 0.030, "c_K": 0.025,
                  "temp_opt": 30, "ph_opt": 6.5},
    "sugarcane": {"yield_max": 70.0, "c_N": 0.010, "c_P": 0.018, "c_K": 0.016,
                  "temp_opt": 30, "ph_opt": 6.0},
    "barley":    {"yield_max": 5.0,  "c_N": 0.025, "c_P": 0.040, "c_K": 0.045,
                  "temp_opt": 16, "ph_opt": 6.5},
}


def simulate_yield(N, P, K, crop_name, temperature, soil_ph):
    """Симуляция урожайности по модели Митчерлиха."""
    p = CROP_PARAMS.get(crop_name, CROP_PARAMS["wheat"])

    n_resp = 1 - np.exp(-p["c_N"] * np.maximum(N, 0))
    p_resp = 1 - np.exp(-p["c_P"] * np.maximum(P, 0))
    k_resp = 1 - np.exp(-p["c_K"] * np.maximum(K, 0))

    temp_f = np.exp(-0.5 * ((temperature - p["temp_opt"]) / 6) ** 2)
    ph_f = np.exp(-0.5 * ((soil_ph - p["ph_opt"]) / 1.2) ** 2)

    return p["yield_max"] * n_resp * p_resp * k_resp * temp_f * ph_f


def yield_simulation_analysis(test_data, predictions, actual_npk, df_info):
    """
    Сравнение урожайности:
    1) При рекомендованных дозах (модель)
    2) При фактических оптимальных дозах
    3) При baseline (средние дозы)
    """
    # Используем средние условия для симуляции
    n_test = len(actual_npk)
    crops = list(CROP_PARAMS.keys())
    temperature = 22.0
    soil_ph = 6.3

    results = {}

    for model_name, pred_npk in predictions.items():
        pred_npk = np.array(pred_npk)

        # Урожай с рекомендациями модели
        yield_model = []
        yield_optimal = []
        yield_baseline = []

        for i in range(min(n_test, len(pred_npk))):
            crop = crops[i % len(crops)]

            # Модель
            y_m = simulate_yield(
                pred_npk[i, 0], pred_npk[i, 1], pred_npk[i, 2],
                crop, temperature, soil_ph
            )
            yield_model.append(y_m)

            # Оптимальные (actual target)
            y_o = simulate_yield(
                actual_npk[i, 0], actual_npk[i, 1], actual_npk[i, 2],
                crop, temperature, soil_ph
            )
            yield_optimal.append(y_o)

            # Baseline: фиксированные средние дозы
            y_b = simulate_yield(100, 50, 50, crop, temperature, soil_ph)
            yield_baseline.append(y_b)

        yield_model = np.array(yield_model)
        yield_optimal = np.array(yield_optimal)
        yield_baseline = np.array(yield_baseline)

        # Метрики
        improvement_vs_baseline = (
            (yield_model.mean() - yield_baseline.mean()) / yield_baseline.mean() * 100
        )
        gap_to_optimal = (
            (yield_optimal.mean() - yield_model.mean()) / yield_optimal.mean() * 100
        )

        # Экономия удобрений vs baseline
        total_model = pred_npk[:n_test].sum(axis=1).mean()
        total_baseline = 200  # 100+50+50
        cost_saving = (total_baseline - total_model) / total_baseline * 100

        results[model_name] = {
            "yield_model_mean": round(float(yield_model.mean()), 3),
            "yield_optimal_mean": round(float(yield_optimal.mean()), 3),
            "yield_baseline_mean": round(float(yield_baseline.mean()), 3),
            "improvement_vs_baseline_%": round(improvement_vs_baseline, 2),
            "gap_to_optimal_%": round(gap_to_optimal, 2),
            "cost_saving_%": round(cost_saving, 2),
        }

    return results


# ========================== ВИЗУАЛИЗАЦИЯ ==========================

def plot_model_comparison(all_metrics, save_path):
    """Столбчатая диаграмма сравнения моделей."""
    models = list(all_metrics.keys())
    targets = ["N", "P", "K"]
    colors = {"Linear": "#e74c3c", "RandomForest": "#3498db",
              "XGBoost": "#f39c12", "MLP": "#2ecc71"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric_name, title in zip(
        axes, ["MAE", "R2", "MAPE"],
        ["MAE (↓ лучше)", "R² (↑ лучше)", "MAPE % (↓ лучше)"]
    ):
        x = np.arange(len(targets))
        width = 0.18

        for i, model in enumerate(models):
            vals = [all_metrics[model].get(t, {}).get(metric_name, 0) for t in targets]
            bars = ax.bar(x + i * width, vals, width, label=model,
                          color=colors.get(model, "gray"), edgecolor="white")

        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(targets)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Сравнение моделей по выходам N, P, K", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Сравнение моделей: {save_path}")


def plot_predictions_scatter(actual, predictions, save_path):
    """Scatter: actual vs predicted для каждой модели."""
    targets = ["N (кг/га)", "P (кг/га)", "K (кг/га)"]
    actual = np.array(actual)
    models = list(predictions.keys())
    n_models = len(models)

    fig, axes = plt.subplots(n_models, 3, figsize=(15, 4 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    colors = {"Linear": "#e74c3c", "RandomForest": "#3498db",
              "XGBoost": "#f39c12", "MLP": "#2ecc71"}

    for row, model_name in enumerate(models):
        pred = np.array(predictions[model_name])
        for col in range(3):
            ax = axes[row, col]
            c = colors.get(model_name, "gray")
            ax.scatter(actual[:, col], pred[:, col], alpha=0.3, s=10, color=c)

            # Линия идеального предсказания
            lim_min = min(actual[:, col].min(), pred[:, col].min())
            lim_max = max(actual[:, col].max(), pred[:, col].max())
            ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.5)

            ax.set_xlabel("Факт")
            ax.set_ylabel("Прогноз")
            ax.set_title(f"{model_name} — {targets[col]}", fontsize=10)
            ax.grid(alpha=0.3)

    plt.suptitle("Actual vs Predicted", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Scatter plots: {save_path}")


def plot_yield_simulation(sim_results, save_path):
    """Визуализация результатов симуляции урожайности."""
    models = list(sim_results.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"Linear": "#e74c3c", "RandomForest": "#3498db",
              "XGBoost": "#f39c12", "MLP": "#2ecc71"}

    # Прирост урожайности vs baseline
    improvements = [sim_results[m]["improvement_vs_baseline_%"] for m in models]
    c = [colors.get(m, "gray") for m in models]
    bars1 = ax1.bar(models, improvements, color=c, edgecolor="white")
    ax1.set_ylabel("Прирост (%)")
    ax1.set_title("Прирост урожая vs baseline (фикс. дозы)", fontweight="bold")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, improvements):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:+.1f}%", ha="center", fontweight="bold", fontsize=10)

    # Экономия удобрений
    savings = [sim_results[m]["cost_saving_%"] for m in models]
    bars2 = ax2.bar(models, savings, color=c, edgecolor="white")
    ax2.set_ylabel("Экономия (%)")
    ax2.set_title("Экономия удобрений vs baseline", fontweight="bold")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, savings):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:+.1f}%", ha="center", fontweight="bold", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Симуляция урожая: {save_path}")


# ========================== MAIN ==========================

def main():
    if not os.path.exists("results/metrics.json"):
        print("Результаты не найдены. Сначала: python train.py")
        sys.exit(1)

    with open("results/metrics.json", encoding="utf-8") as f:
        data = json.load(f)

    with open("results/predictions.json") as f:
        preds_data = json.load(f)

    all_metrics = data["metrics"]
    actual = np.array(preds_data["actual"])
    predictions = {k: np.array(v) for k, v in preds_data.items() if k != "actual"}

    # 1. Таблица метрик
    print(f"\n{'='*70}")
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print(f"{'='*70}")

    targets = ["N", "P", "K"]
    print(f"\n{'Модель':<15} {'N MAE':>7} {'P MAE':>7} {'K MAE':>7} "
          f"{'Avg R²':>7} {'Avg MAPE':>9}")
    print("-" * 60)

    sorted_models = sorted(
        all_metrics.items(),
        key=lambda x: x[1].get("AVG", {}).get("MAPE", 100)
    )

    for name, m in sorted_models:
        n_mae = m.get("N", {}).get("MAE", 0)
        p_mae = m.get("P", {}).get("MAE", 0)
        k_mae = m.get("K", {}).get("MAE", 0)
        avg_r2 = m.get("AVG", {}).get("R2", 0)
        avg_mape = m.get("AVG", {}).get("MAPE", 0)
        medal = "🥇" if name == sorted_models[0][0] else "  "
        print(f"{medal}{name:<13} {n_mae:>7.2f} {p_mae:>7.2f} {k_mae:>7.2f} "
              f"{avg_r2:>7.4f} {avg_mape:>8.1f}%")

    # 2. Графики
    plot_model_comparison(all_metrics, "results/model_comparison.png")
    plot_predictions_scatter(actual, predictions, "results/scatter_plots.png")

    # 3. Симуляция урожайности
    print(f"\n{'='*70}")
    print("СИМУЛЯЦИЯ УРОЖАЙНОСТИ")
    print(f"{'='*70}")

    sim_results = yield_simulation_analysis(None, predictions, actual, data)

    print(f"\n{'Модель':<15} {'Урожай':>8} {'vs Baseline':>12} "
          f"{'Gap to Opt':>12} {'Экономия':>10}")
    print("-" * 62)

    for name, r in sim_results.items():
        print(f"{name:<15} {r['yield_model_mean']:>8.3f} "
              f"{r['improvement_vs_baseline_%']:>+11.1f}% "
              f"{r['gap_to_optimal_%']:>11.1f}% "
              f"{r['cost_saving_%']:>+9.1f}%")

    plot_yield_simulation(sim_results, "results/yield_simulation.png")

    # 4. Сохранение отчёта
    report = {
        "model_metrics": all_metrics,
        "yield_simulation": sim_results,
        "best_model": sorted_models[0][0],
    }

    with open("results/full_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open("results/report.txt", "w", encoding="utf-8") as f:
        f.write("ОТЧЁТ: Рекомендация удобрений NPK\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Лучшая модель: {sorted_models[0][0]}\n\n")
        f.write("Метрики:\n")
        for name, m in sorted_models:
            avg = m.get("AVG", {})
            f.write(f"  {name}: MAE={avg.get('MAE',0):.2f}, "
                    f"R²={avg.get('R2',0):.4f}, MAPE={avg.get('MAPE',0):.1f}%\n")
        f.write("\nСимуляция урожайности:\n")
        for name, r in sim_results.items():
            f.write(f"  {name}: прирост={r['improvement_vs_baseline_%']:+.1f}%, "
                    f"экономия={r['cost_saving_%']:+.1f}%\n")

    print(f"\nОтчёт: results/report.txt")
    print(f"JSON: results/full_report.json")


if __name__ == "__main__":
    main()
