import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# =============================================================================
# 1. 安全匯入 (請確保與 main_pipeline 等檔案在同一目錄)
# =============================================================================
from main_pipeline import (
    model3_dynamic_adjustment,
    get_model2_recommendations,
    HAS_TEAM_CODE,
    country_mapping,
    lang_mapping
)

if HAS_TEAM_CODE:
    from model1_LSM import predict_all_colours
    from model2_classification import set_binary_category_input
else:
    predict_all_colours = None
    set_binary_category_input = None

# =============================================================================
# 2. 獨立的攔截器迴圈 (專注對比「退火前」vs「退火後」)
# =============================================================================
def run_annealing_isolation_test(user_profile, main_color, raw_user_emotions, confidence_threshold=0.30):
    # 取得 Model 1 的基準分數
    if HAS_TEAM_CODE:
        all_m1 = predict_all_colours(user_profile, params_dir='model1_params')
        m1_scores = all_m1.get(main_color, {"emotion_vitality": 1.0, "emotion_stability": 1.0, "emotion_resonance": 1.0, "emotion_alert": 1.0})
    else:
        m1_scores = {"emotion_vitality": 0.036, "emotion_stability": 0.251, "emotion_resonance": 1.246, "emotion_alert": 1.205}
    
    current_emotions = raw_user_emotions.copy()
    max_attempts = 3
    
    prob_before_annealing = None
    prob_after_annealing = None
    
    for attempt in range(1, max_attempts + 1):
        # 注意：動態權重在這裡「每一次」都會被保留，我們只測試退火的影響
        adjusted_emotions = model3_dynamic_adjustment(m1_scores, current_emotions)
        user_input_m2 = adjusted_emotions.copy()
        user_input_m2.update({"fluentenglish": user_profile["fluentenglish"], "gender": user_profile["gender"], "age_group": user_profile["age_group"]})
        
        # 國家/語言編碼轉換
        if HAS_TEAM_CODE and country_mapping is not None and lang_mapping is not None:
            user_input_m2 = set_binary_category_input(user_input_m2, country_mapping, "residencecountry", "country_binary", "country", user_profile.get("country_code", "us"))
            user_input_m2 = set_binary_category_input(user_input_m2, lang_mapping, "mothertongue", "lang_binary", "lang", user_profile.get("lang_code", "en"))
        
        # 取得模型二推薦結果
        recommendations = get_model2_recommendations(user_input_m2)
        top1_prob = recommendations[0]["prob"]
        
        # 🔥 攔截點 1：紀錄第一次預測 (退火啟動前)
        if attempt == 1:
            prob_before_annealing = top1_prob
            
        # 🔥 攔截點 2：不斷更新最後一次預測 (退火完成後)
        prob_after_annealing = top1_prob
        
        # 退火閾值判定
        if top1_prob >= confidence_threshold:
            break
        else:
            # 執行退火：向常態中位數 2.5 靠攏
            for k in current_emotions.keys():
                current_emotions[k] = current_emotions[k] * 0.9 + 2.5 * 0.1
            
    return prob_before_annealing, prob_after_annealing

# =============================================================================
# 3. 啟動實驗並繪圖
# =============================================================================
def main():
    print("🚀 啟動退火機制孤立測試 (控制變因純淨版) ...")
    n_tests = 1000
    np.random.seed(42)
    colors = ['black', 'grey', 'white', 'brown'] 
    
    probs_before = []
    probs_after = []
    
    # 🔥 將使用者特徵鎖死 (控制變因)，消除族群差異帶來的雜訊
    fixed_user_profile = {
        "fluentenglish": 8,
        "gender": 1,             # 男性
        "age_group": 1,          # 18~35歲
        "country_code": "tw",    # 台灣
        "lang_code": "zh"        # 中文
    }
    
    # 生成 1000 位極端奧客進行實測
    for _ in tqdm(range(n_tests), desc="測試進度"):
        main_color = np.random.choice(colors)
        
        # 唯一變動的只有輸入的情緒與選擇的顏色
        raw_user_emotions = {
            "emotion_vitality": np.random.choice([0.0, 5.0]),
            "emotion_stability": np.random.choice([0.0, 5.0]),
            "emotion_resonance": np.random.choice([0.0, 5.0]),
            "emotion_alert": np.random.choice([0.0, 5.0])
        }
        
        before, after = run_annealing_isolation_test(fixed_user_profile, main_color, raw_user_emotions, confidence_threshold=0.30)
        probs_before.append(before)
        probs_after.append(after)
        
    probs_before = np.array(probs_before) * 100
    probs_after = np.array(probs_after) * 100
    
    # 統計數據
    print("\n📊 1,000 次真實極端壓力測試結果 (固定使用者背景)：")
    print(f" - [退火前 (第1次)] 平均信心度: {np.mean(probs_before):.1f}% | 危險盲猜率 (<12%): {np.sum(probs_before < 12)/n_tests*100:.1f}%")
    print(f" - [退火後 (最終)]   平均信心度: {np.mean(probs_after):.1f}% | 危險盲猜率 (<12%): {np.sum(probs_after < 12)/n_tests*100:.1f}%")

    # 繪製學術圖表
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    data = [probs_before, probs_after]
    labels = ['退火前 (Attempt 1)', '退火後 (Final Attempt)']

    plt.figure(figsize=(9, 6))
    sns.boxplot(data=data, palette=['#e74c3c', '#2ecc71'], width=0.5, showfliers=False)
    sns.stripplot(data=data, color=".3", size=3, alpha=0.3, jitter=True)

    plt.axhline(y=12, color='gray', linestyle='--', label="盲猜警戒線 (12%)")
    plt.axhline(y=30, color='red', linestyle=':', label="退火目標閾值 (30%)")

    plt.xticks([0, 1], labels, fontsize=14)
    plt.ylabel("Top-1 預測信心度 (%)", fontsize=14)
    plt.title("真實模型退火機制測試 (控制族群變因, N=1,000)", fontsize=18, fontweight='bold', pad=15)
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("real_annealing_fixed_user.png", dpi=300)
    print("✅ 測試完成！純淨數據圖表已儲存為 real_annealing_fixed_user.png")
    plt.show()

if __name__ == "__main__":
    main()