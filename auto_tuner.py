import numpy as np
import itertools
from collections import Counter
import random
import time

print("🚀 啟動 SBTI 權重自動優化引擎 (Auto-Tuner)...")
time.sleep(1)

# ==========================================
# 1. 你的大數據矩陣
# ==========================================
MODEL1_COLOR_MATRIX = {
    "black": [0.1561, 0.2815, 1.2825, 1.2854],
    "blue": [0.7046, 0.9963, 0.4500, 0.1430],
    "brown": [0.1349, 0.2523, 0.4796, 0.5289],
    "green": [0.9329, 0.9181, 0.1524, 0.2344],
    "grey": [0.1044, 0.2313, 1.1857, 0.4604],
    "orange": [1.1481, 0.5428, 0.1397, 0.2081],
    "pink": [1.4019, 0.6664, 0.1154, 0.0870],
    "purple": [0.6928, 0.6554, 0.3996, 0.2601],
    "red": [1.3358, 0.4869, 0.2971, 0.9091],
    "turquoise": [0.9576, 0.8634, 0.1382, 0.0817],
    "white": [0.6105, 1.1301, 0.2377, 0.1326],
    "yellow": [1.2555, 0.6491, 0.1878, 0.2600],
}


def c(color): return np.array(MODEL1_COLOR_MATRIX[color])

# ==========================================
# 2. 自動搜尋引擎設定
# ==========================================
ITERATIONS = 1000  # 測試 1000 種配分組合
best_std = float('inf') # 標準差越小代表越平均
best_weights = {}
best_distribution = []

print(f"🔍 正在暴力搜尋 {ITERATIONS} 種配分平行宇宙，尋找最平均的解...\n")

for i in range(ITERATIONS):
    # 隨機生成各種權重 (為符合邏輯，主要顏色權重 > 次要顏色)
    w1_m, w1_s = random.randint(4, 8), random.randint(1, 3)
    w2_pure = random.randint(1, 3) # 純色壓低
    w3_pure = random.randint(1, 3)
    w4_m, w4_s = random.randint(3, 7), random.randint(1, 3)
    w5_m, w5_s = random.randint(5, 10), random.randint(2, 4) # Q5放最大
    
    # 建立這組權重的題目選項
    q1 = [
        w1_m*c("black") + w1_m*c("grey") + w1_s*c("blue") + w1_s*c("white"),
        w1_m*c("yellow") + w1_m*c("orange") + w1_s*c("red") + w1_s*c("turquoise"),
        w1_m*c("brown") + w1_m*c("white") + w1_s*c("pink") + w1_s*c("green"),
        w1_m*c("blue") + w1_m*c("white") + w1_s*c("black") + w1_s*c("grey")
    ]
    q2 = [w2_pure*c(color) for color in MODEL1_COLOR_MATRIX.keys()]
    q3 = [w3_pure*c(color) for color in MODEL1_COLOR_MATRIX.keys()]
    q4 = [
        w4_m*c("brown") + w4_m*c("green") + w4_s*c("grey") + w4_s*c("white"),
        w4_m*c("black") + w4_m*c("blue") + w4_s*c("purple") + w4_s*c("turquoise"),
        w4_m*c("grey") + w4_m*c("white") + w4_s*c("black") + w4_s*c("blue"),
        w4_m*c("orange") + w4_m*c("yellow") + w4_s*c("brown") + w4_s*c("green"),
        w4_m*c("pink") + w4_m*c("purple") + w4_s*c("red") + w4_s*c("orange"),
        w4_m*c("red") + w4_m*c("turquoise") + w4_s*c("pink") + w4_s*c("yellow")
    ]
    q5 = [
        w5_m*c("blue") + w5_m*c("yellow") + w5_s*c("turquoise") + w5_s*c("orange"),
        w5_m*c("black") + w5_m*c("red") + w5_s*c("grey") + w5_s*c("purple"),
        w5_m*c("green") + w5_m*c("turquoise") + w5_s*c("blue") + w5_s*c("white"),
        w5_m*c("pink") + w5_m*c("white") + w5_s*c("red") + w5_s*c("yellow"),
        w5_m*c("brown") + w5_m*c("orange") + w5_s*c("green") + w5_s*c("yellow"),
        w5_m*c("purple") + w5_m*c("grey") + w5_s*c("black") + w5_s*c("pink")
    ]

    # 計算 20736 種組合的中位數
    all_totals = [q1_v + q2_v + q3_v + q4_v + q5_v for q1_v in q1 for q2_v in q2 for q3_v in q3 for q4_v in q4 for q5_v in q5]
    threshold = np.median(all_totals, axis=0)

    # 統計結果
    results = []
    for total_score in all_totals:
        v = "H" if total_score[0] >= threshold[0] else "L"
        s = "H" if total_score[1] >= threshold[1] else "L"
        r = "H" if total_score[2] >= threshold[2] else "L"
        a = "H" if total_score[3] >= threshold[3] else "L"
        results.append(f"{v}{s}{r}{a}")

    counts = list(Counter(results).values())
    
    # 補齊沒出現的人格 (如果有0次的話)
    while len(counts) < 16:
        counts.append(0)
        
    # 計算這組分配的標準差 (Standard Deviation)
    # 標準差越小，代表這 16 個人格的分佈越平均！
    current_std = np.std(counts)

    if current_std < best_std:
        best_std = current_std
        best_weights = {
            "Q1(主/次)": (w1_m, w1_s),
            "Q2(純色)": w2_pure,
            "Q3(純色)": w3_pure,
            "Q4(主/次)": (w4_m, w4_s),
            "Q5(主/次)": (w5_m, w5_s)
        }
        best_distribution = Counter(results).most_common()

# ==========================================
# 3. 印出最佳結果
# ==========================================
print("🎉 搜尋完成！找到最能打破極端分佈的神級配分！\n")
print("🔥 【最佳權重配置】請將這些數字填回你的 app.py：")
for k, v in best_weights.items():
    print(f"  - {k}: {v}")

print("\n📊 【優化後的 16 型人格機率】")
for code, count in best_distribution:
    percentage = (count / 20736) * 100
    print(f"  {code} : {count:5d} 次  ({percentage:5.2f}%)")
