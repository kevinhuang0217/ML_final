import numpy as np
import itertools
from collections import Counter

# ==========================================
# 1. 貼上你們真實的 MODEL1_COLOR_MATRIX
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

# 輔助函式，方便下面抓取陣列
def c(color): return np.array(MODEL1_COLOR_MATRIX[color])

# ==========================================
# 2. 五道題目的選項權重分數 
# (🔥 注意這裡！Q2 和 Q3 的權重已經從 8 降為 4)
# ==========================================
q1 = [
    3*c("black") + 3*c("grey") + 1*c("blue") + 1*c("white"),
    3*c("yellow") + 3*c("orange") + 1*c("red") + 1*c("turquoise"),
    3*c("brown") + 3*c("white") + 1*c("pink") + 1*c("green"),
    3*c("blue") + 3*c("white") + 1*c("black") + 1*c("grey")
]

# 👇 權重稀釋：將純色題權重降為 4，打破極端情緒的綁定
q2 = [4*c(color) for color in MODEL1_COLOR_MATRIX.keys()]
q3 = [4*c(color) for color in MODEL1_COLOR_MATRIX.keys()]

q4 = [
    3*c("brown") + 3*c("green") + 1*c("grey") + 1*c("white"),
    3*c("black") + 3*c("blue") + 1*c("purple") + 1*c("turquoise"),
    3*c("grey") + 3*c("white") + 1*c("black") + 1*c("blue"),
    3*c("orange") + 3*c("yellow") + 1*c("brown") + 1*c("green"),
    3*c("pink") + 3*c("purple") + 1*c("red") + 1*c("orange"),
    3*c("red") + 3*c("turquoise") + 1*c("pink") + 1*c("yellow")
]
q5 = [
    3*c("blue") + 3*c("yellow") + 1*c("turquoise") + 1*c("orange"),
    3*c("black") + 3*c("red") + 1*c("grey") + 1*c("purple"),
    3*c("green") + 3*c("turquoise") + 1*c("blue") + 1*c("white"),
    3*c("pink") + 3*c("white") + 1*c("red") + 1*c("yellow"),
    3*c("brown") + 3*c("orange") + 1*c("green") + 1*c("yellow"),
    3*c("purple") + 3*c("grey") + 1*c("black") + 1*c("pink")
]

# ==========================================
# 3. 展開 20,736 種組合並計算「中位數閾值」
# ==========================================
all_combinations = list(itertools.product(q1, q2, q3, q4, q5))
all_totals = [sum(combo) for combo in all_combinations]

# 計算中位數 (保證完美 50/50 的切分點)
threshold = np.median(all_totals, axis=0)

# ==========================================
# 4. 統計 16 種人格出現次數
# ==========================================
results = []
for total_score in all_totals:
    v = "H" if total_score[0] >= threshold[0] else "L"
    s = "H" if total_score[1] >= threshold[1] else "L"
    r = "H" if total_score[2] >= threshold[2] else "L"
    a = "H" if total_score[3] >= threshold[3] else "L"
    results.append(f"{v}{s}{r}{a}")

counter = Counter(results)
total_count = len(results)

# ==========================================
# 5. 印出結果表格
# ==========================================
print("\n" + "="*55)
print(f"🔥 權重稀釋後之中位數閾值 [V, S, R, A] = {np.round(threshold, 2)}")
print(f"📊 總共模擬了 {total_count} 種作答組合")
print("="*55)
print("👑 16 型人格出現機率分佈 (由高到低)：\n")

for code, count in counter.most_common():
    percentage = (count / total_count) * 100
    print(f"  {code} : {count:5d} 次  ({percentage:5.2f}%)")
print("="*55 + "\n")