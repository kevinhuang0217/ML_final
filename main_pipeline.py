# -*- coding: utf-8 -*-
"""
【機器學習期末專題 — 總主程式 Pipeline Master (終極科學版)】
包含：Model 1 (LSM) -> Model 3 (動態權重壓制) -> Model 2 (手工高斯貝氏分類器 + 退火迴圈)
特色：
1. 完全移除人為主觀防呆，100% 依賴資料真實分佈。
2. 不對 Model 1 進行造假放大，採用「動態權重壓制 (Alpha Decay)」處理尺度差異。
3. 具備 Doubt Option 低信心退火機制，確保系統極高的魯棒性 (Robustness)。
"""
import numpy as np
import pandas as pd
import os

# =============================================================================
# 🔗 模組匯入區 (對接真實的 Model 1 與 Model 2)
# =============================================================================
try:
    from model1_LSM import predict_all_colours
    from model2_classification import set_binary_category_input, model2_predict_top3, feature_cols
    HAS_TEAM_CODE = True
    print("✅ 成功載入 Model 1 與 Model 2 的真實預測模型！")
except ImportError as e:
    HAS_TEAM_CODE = False
    print(f"⚠️ 警告：無法載入組員的模型檔案 ({e})。系統將切換至模擬模式。")

# 載入二進位映射表
if os.path.exists("country_binary_mapping.csv") and os.path.exists("lang_binary_mapping.csv"):
    country_mapping = pd.read_csv("country_binary_mapping.csv")
    lang_mapping = pd.read_csv("lang_binary_mapping.csv")
else:
    country_mapping, lang_mapping = None, None

# =============================================================================
# 📈 模型三：動態邊界錨定演算法 (權重壓制版 Alpha Decay)
# =============================================================================
def model3_dynamic_adjustment(m1_color_scores, user_desired_scores):
    """
    不篡改 Model 1 的真實尺度，純粹透過計算絕對差距(Delta)來壓制使用者的權重。
    當使用者要求遠超色彩本質時，將權重降至極低 (e.g., 0.1)，達成完美的先驗錨定。
    """
    emotion_keys = ["emotion_vitality", "emotion_stability", "emotion_resonance", "emotion_alert"]
    adjusted_scores = {}
    
    for key in emotion_keys:
        m1_val = m1_color_scores.get(key, 0.0)
        u_val = user_desired_scores.get(key, 0.0)
        
        # 在保留真實數據尺度的前提下計算差距
        delta = abs(u_val - m1_val)
        
        # 動態 Alpha (使用者權重) 壓制邏輯
        if delta > 3.5:
            alpha = 0.1  # 嚴重衝突：壓制使用者，只聽使用者 10%，主色佔 90%
        elif delta > 2.0:
            alpha = 0.5  # 中等衝突：雙方各佔 50%
        else:
            alpha = 0.8  # 差距極小：高度信任使用者意圖，佔 80%
            
        # 採用嚴謹的「加權平均法 (Weighted Interpolation)」
        new_score = (u_val * alpha) + (m1_val * (1.0 - alpha))
        
        # 確保最後結果不低於0，也不超過5
        adjusted_scores[key] = max(0.0, min(5.0, new_score))
        
    return adjusted_scores

# =============================================================================
# 🧠 模型二：真實貝氏分類器預測層
# =============================================================================
def get_model2_recommendations(user_input_m2):
    if not HAS_TEAM_CODE:
        return [{"colour": "yellow", "prob": 0.45}, {"colour": "red", "prob": 0.25}, {"colour": "orange", "prob": 0.15}]
    
    final_input = {col: 0 for col in feature_cols}
    final_input.update(user_input_m2)
    
    # 💡 修正Bug：向模型索取 Top-5，確保過濾主色後依然有足夠的顏色
    raw_recommendations = model2_predict_top3(input_dict=final_input, top_k=5)
    
    recommendations = []
    for rec in raw_recommendations:
        recommendations.append({
            "colour": rec["colour"],
            "prob": float(rec["posterior_probability"])
        })
        
    return recommendations

# =============================================================================
# 🎛️ 系統總運作流水線 (⚠️ 消融實驗版：拔除動態權重，寫死 50/50)
# =============================================================================
def run_advanced_pipeline(user_profile, main_color, raw_user_emotions, confidence_threshold=0.30):
    # 1. 取得 Model 1 的真實大數據基準
    if HAS_TEAM_CODE:
        all_m1 = predict_all_colours(user_profile, params_dir='model1_params')
        m1_scores = all_m1.get(main_color, {"emotion_vitality": 1.0, "emotion_stability": 1.0, "emotion_resonance": 1.0, "emotion_alert": 1.0})
    else:
        m1_scores = {"emotion_vitality": 0.036, "emotion_stability": 0.251, "emotion_resonance": 1.246, "emotion_alert": 1.205}
    
    # =================================================================
    # 💥 【拔掉保護罩】把你的動態權重註解掉，改成無腦 50/50
    # =================================================================
    # 原本的完美版：
    # adjusted_emotions = model3_dynamic_adjustment(m1_scores, raw_user_emotions)
    
    # 現在的災難版 (強制對切平分)：
    adjusted_emotions = {}
    for k in m1_scores.keys():
        user_val = raw_user_emotions.get(k, 0)
        adjusted_emotions[k] = (m1_scores[k] + user_val) / 2.0
    # =================================================================
        
    current_emotions = adjusted_emotions.copy()
    
    # 3. 啟動導航儀：退火迴圈 (最多 3 次) - 這邊保留，讓你看看即使有退火也救不回來的慘況
    max_iterations = 3
    final_recs = []
    
    for i in range(max_iterations + 1):
        # 組合 Model 2 需要的特徵
        user_input_m2 = current_emotions.copy()
        user_input_m2.update({"fluentenglish": user_profile["fluentenglish"], "gender": user_profile["gender"], "age_group": user_profile["age_group"]})
        
        if HAS_TEAM_CODE and country_mapping is not None and lang_mapping is not None:
            user_input_m2 = set_binary_category_input(user_input_m2, country_mapping, "residencecountry", "country_binary", "country", user_profile.get("country_code", "us"))
            user_input_m2 = set_binary_category_input(user_input_m2, lang_mapping, "mothertongue", "lang_binary", "lang", user_profile.get("lang_code", "en"))
        
        # 丟給分類器預測
        recommendations = get_model2_recommendations(user_input_m2)
        
        # 嚴格過濾掉主色
        filtered_recs = [r for r in recommendations if r["colour"] != main_color]
        
        if not filtered_recs:
            break
            
        top1_prob = filtered_recs[0]["prob"]
        final_recs = filtered_recs[:3] # 取前三名
        
        # 判斷是否需要退火
        if top1_prob >= confidence_threshold or i == max_iterations:
            break
            
        # 退火公式
        for k in current_emotions:
            current_emotions[k] = current_emotions[k] * 0.8 + 2.5 * 0.2

    return final_recs

# =============================================================================
# 🎮 互動式終端機測試介面 (CLI)
# =============================================================================
def get_float_input(prompt_text):
    while True:
        try:
            val = float(input(prompt_text))
            if 0.0 <= val <= 5.0: return val
            print("  ❌ 請輸入 0.0 到 5.0 之間的數字！")
        except ValueError:
            print("  ❌ 請輸入有效的數字！")

if __name__ == "__main__":
    print("\n" + "🌟"*25)
    print(" 歡迎來到【大學生簡報配色推薦系統】後台測試")
    print("🌟"*25)
    
    valid_colors = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'turquoise', 'white', 'yellow']
    print("\n🎨 可選的主色有：")
    print(" | ".join(valid_colors))
    
    while True:
        main_color = input("\n👉 請輸入你的簡報主色: ").strip().lower()
        if main_color in valid_colors:
            break
        print(f"  ❌ 找不到這個顏色！可選顏色: {', '.join(valid_colors)}")

    print("\n🎛️ 請設定你期望簡報給人的感受 (0.0 ~ 5.0)：")
    v = get_float_input("   🏃 活力感 (Vitality) : ")
    s = get_float_input("   🧘 穩定感 (Stability): ")
    r = get_float_input("   💧 共鳴感 (Resonance): ")
    a = get_float_input("   ⚠️ 警示感 (Alert)    : ")
    
    user_emotions = {
        "emotion_vitality": v,
        "emotion_stability": s,
        "emotion_resonance": r,
        "emotion_alert": a
    }
    
    print("\n👥 (載入預設使用者背景: 女性, 青年, 英語流利度 8, 居住美國, 英文母語)")
    user_profile = {"gender": 0, "age_group": 1, "fluentenglish": 8, "country_code": "us", "lang_code": "en"}
    
    input("\n✅ 設定完成！請按 [Enter] 鍵啟動真實機器學習引擎...")
    
    run_advanced_pipeline(user_profile, main_color, user_emotions)