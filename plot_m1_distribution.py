import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 設定中文字型 (避免 matplotlib 中文亂碼)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 模擬 Model 1 的真實大數據預測輸出 (根據線性迴歸的特性，通常呈常態分佈且變異數較小)
# 假設我們對 10,000 個不同受眾跑了 Model 1 的「活力值」預測
np.random.seed(42)
m1_predictions = np.random.normal(loc=2.5, scale=0.6, size=10000)
# 確保數值在 0~5 之間
m1_predictions = np.clip(m1_predictions, 0, 5)

# 2. 開始畫圖
plt.figure(figsize=(10, 6))
sns.kdeplot(m1_predictions, fill=True, color="#3867d6", alpha=0.5, linewidth=2)

# 3. 畫上極端使用者的輸入點 (5.0)
plt.axvline(x=5.0, color='#d9363e', linestyle='--', linewidth=2)
plt.scatter(5.0, 0, color='#d9363e', s=200, zorder=5, label="使用者輸入的極端值 (5.0)")

# 4. 畫上動態權重拉回的點 (0.1 權重)
adjusted_value = 5.0 * 0.1 + 2.5 * 0.9  # 假設 M1 平均為 2.5，被 alpha=0.1 拉回
plt.axvline(x=adjusted_value, color='#2f8f5b', linestyle='-.', linewidth=2)
plt.scatter(adjusted_value, 0, color='#2f8f5b', s=200, zorder=5, label=f"動態權重拉回後的安全值 ({adjusted_value:.2f})")

# 5. 設定標籤與排版
plt.title("Model 1 預測分數分佈 vs 使用者極端輸入", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("情緒分數 (0.0 ~ 5.0)", fontsize=14)
plt.ylabel("資料分佈密度 (Probability Density)", fontsize=14)
plt.xlim(0, 5.5)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("m1_distribution_proof.png", dpi=300)
print("✅ 圖表已儲存為 m1_distribution_proof.png")
plt.show()