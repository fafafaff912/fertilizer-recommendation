"""
Рекомендация удобрений для конкретного поля.

Запуск:
    python predict.py --temperature 25 --humidity 60 --rainfall 120 \
        --soil_ph 6.5 --soil_N 30 --soil_P 20 --soil_K 25 \
        --crop wheat --prev_yield 3.5

    python predict.py --interactive
"""

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# === Модель MLP (копия из train.py) ===
class FertilizerMLP(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# === Параметры культур для симуляции ===
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


def simulate_yield(N, P, K, crop, temp, ph):
    p = CROP_PARAMS.get(crop, CROP_PARAMS["wheat"])
    n_r = 1 - np.exp(-p["c_N"] * max(N, 0))
    p_r = 1 - np.exp(-p["c_P"] * max(P, 0))
    k_r = 1 - np.exp(-p["c_K"] * max(K, 0))
    t_f = np.exp(-0.5 * ((temp - p["temp_opt"]) / 6) ** 2)
    ph_f = np.exp(-0.5 * ((ph - p["ph_opt"]) / 1.2) ** 2)
    return p["yield_max"] * n_r * p_r * k_r * t_f * ph_f


def predict_single(args):
    """Рекомендация для одного поля."""
    device = torch.device("cpu")

    # Загрузка MLP
    mlp_path = "results/mlp_model.pth"
    sklearn_path = "results/sklearn_models.pkl"

    if not os.path.exists(mlp_path) or not os.path.exists(sklearn_path):
        print("Модели не найдены. Сначала: python train.py")
        sys.exit(1)

    checkpoint = torch.load(mlp_path, map_location=device, weights_only=False)
    with open(sklearn_path, "rb") as f:
        sk_data = pickle.load(f)

    le = checkpoint["label_encoder"]
    scaler_X = checkpoint["scaler_X"]
    scaler_y = checkpoint["scaler_y"]
    input_dim = checkpoint["input_dim"]

    # Кодирование культуры
    crop = args.crop.lower()
    if crop not in le.classes_:
        print(f"Неизвестная культура: {crop}")
        print(f"Доступные: {', '.join(le.classes_)}")
        sys.exit(1)

    crop_encoded = le.transform([crop])[0]

    # Формирование вектора признаков
    features = np.array([[
        args.temperature, args.humidity, args.rainfall,
        args.soil_ph, args.soil_moisture,
        args.soil_N, args.soil_P, args.soil_K,
        args.organic_carbon, args.prev_yield,
        args.area, crop_encoded,
    ]], dtype=np.float32)

    features_scaled = scaler_X.transform(features)

    # === Прогнозы всех моделей ===
    results = {}

    # MLP
    model = FertilizerMLP(input_dim, 3).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        pred_s = model(torch.FloatTensor(features_scaled)).numpy()
    pred_mlp = scaler_y.inverse_transform(pred_s).flatten()
    pred_mlp = np.maximum(pred_mlp, 0)
    results["MLP"] = pred_mlp

    # Sklearn модели
    for name, key in [("Linear", "linear"), ("RandomForest", "rf"), ("XGBoost", "xgb")]:
        m = sk_data.get(key)
        if m is not None:
            inp = features_scaled if name == "Linear" else features
            pred = m.predict(inp).flatten()
            pred = np.maximum(pred, 0)
            results[name] = pred

    # === Вывод ===
    print(f"\n{'='*55}")
    print("РЕКОМЕНДАЦИЯ УДОБРЕНИЙ")
    print(f"{'='*55}")
    print(f"\n  Культура:     {crop}")
    print(f"  Температура:  {args.temperature}°C")
    print(f"  Влажность:    {args.humidity}%")
    print(f"  Осадки:       {args.rainfall} мм")
    print(f"  pH почвы:     {args.soil_ph}")
    print(f"  N в почве:    {args.soil_N} кг/га")
    print(f"  P в почве:    {args.soil_P} кг/га")
    print(f"  K в почве:    {args.soil_K} кг/га")
    print(f"  Пред. урожай: {args.prev_yield} т/га")

    print(f"\n{'Модель':<15} {'N (кг/га)':>10} {'P (кг/га)':>10} {'K (кг/га)':>10} "
          f"{'Ож. урожай':>12}")
    print("-" * 62)

    for name, pred in results.items():
        expected_yield = simulate_yield(
            args.soil_N + pred[0], args.soil_P + pred[1], args.soil_K + pred[2],
            crop, args.temperature, args.soil_ph
        )
        print(f"{name:<15} {pred[0]:>10.1f} {pred[1]:>10.1f} {pred[2]:>10.1f} "
              f"{expected_yield:>11.2f} т/га")

    # Baseline (без дополнительных удобрений)
    baseline_yield = simulate_yield(
        args.soil_N, args.soil_P, args.soil_K,
        crop, args.temperature, args.soil_ph
    )
    print(f"{'Без удобрений':<15} {'0':>10s} {'0':>10s} {'0':>10s} "
          f"{baseline_yield:>11.2f} т/га")

    # Фиксированные средние дозы
    fixed_yield = simulate_yield(
        args.soil_N + 100, args.soil_P + 50, args.soil_K + 50,
        crop, args.temperature, args.soil_ph
    )
    print(f"{'Фикс.(100/50/50)':<15} {'100':>10s} {'50':>10s} {'50':>10s} "
          f"{fixed_yield:>11.2f} т/га")

    # === Визуализация ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Рекомендованные дозы
    model_names = list(results.keys())
    x = np.arange(3)
    width = 0.18
    labels_npk = ["N", "P", "K"]
    colors = {"MLP": "#2ecc71", "Linear": "#e74c3c",
              "RandomForest": "#3498db", "XGBoost": "#f39c12"}

    for i, name in enumerate(model_names):
        vals = results[name]
        ax1.bar(x + i * width, vals, width, label=name,
                color=colors.get(name, "gray"), edgecolor="white")

    ax1.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax1.set_xticklabels(labels_npk, fontsize=12)
    ax1.set_ylabel("Доза (кг/га)")
    ax1.set_title(f"Рекомендации для {crop}", fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Ожидаемый урожай
    yields = {}
    for name, pred in results.items():
        yields[name] = simulate_yield(
            args.soil_N + pred[0], args.soil_P + pred[1], args.soil_K + pred[2],
            crop, args.temperature, args.soil_ph
        )
    yields["Без удобр."] = baseline_yield
    yields["Фикс. дозы"] = fixed_yield

    bars = ax2.bar(yields.keys(), yields.values(),
                   color=[colors.get(n, "#95a5a6") for n in yields.keys()],
                   edgecolor="white")
    ax2.set_ylabel("Урожайность (т/га)")
    ax2.set_title("Ожидаемый урожай", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")

    for bar, val in zip(bars, yields.values()):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    save_path = "results/recommendation.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nГрафик: {save_path}")
    plt.show()


def interactive_mode():
    """Интерактивный ввод параметров поля."""
    print("\n🌾 Интерактивная рекомендация удобрений")
    print("=" * 40)

    crops = list(CROP_PARAMS.keys())
    print(f"Доступные культуры: {', '.join(crops)}")

    args = argparse.Namespace()
    args.crop = input("Культура: ").strip() or "wheat"
    args.temperature = float(input("Температура (°C) [25]: ") or 25)
    args.humidity = float(input("Влажность (%) [60]: ") or 60)
    args.rainfall = float(input("Осадки (мм) [120]: ") or 120)
    args.soil_ph = float(input("pH почвы [6.5]: ") or 6.5)
    args.soil_moisture = float(input("Влажность почвы (%) [40]: ") or 40)
    args.soil_N = float(input("N в почве (кг/га) [30]: ") or 30)
    args.soil_P = float(input("P в почве (кг/га) [20]: ") or 20)
    args.soil_K = float(input("K в почве (кг/га) [25]: ") or 25)
    args.organic_carbon = float(input("Орган. углерод (%) [1.5]: ") or 1.5)
    args.prev_yield = float(input("Прошлый урожай (т/га) [3.5]: ") or 3.5)
    args.area = float(input("Площадь (га) [10]: ") or 10)

    predict_single(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Рекомендация удобрений для поля")
    parser.add_argument("--interactive", action="store_true", help="Интерактивный режим")
    parser.add_argument("--crop", type=str, default="wheat")
    parser.add_argument("--temperature", type=float, default=25)
    parser.add_argument("--humidity", type=float, default=60)
    parser.add_argument("--rainfall", type=float, default=120)
    parser.add_argument("--soil_ph", type=float, default=6.5)
    parser.add_argument("--soil_moisture", type=float, default=40)
    parser.add_argument("--soil_N", type=float, default=30)
    parser.add_argument("--soil_P", type=float, default=20)
    parser.add_argument("--soil_K", type=float, default=25)
    parser.add_argument("--organic_carbon", type=float, default=1.5)
    parser.add_argument("--prev_yield", type=float, default=3.5)
    parser.add_argument("--area", type=float, default=10)
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    else:
        predict_single(args)
