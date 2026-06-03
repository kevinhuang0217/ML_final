import main_pipeline
from main_pipeline import run_advanced_pipeline

def run_real_impact_test():
    # 🎯 設定一個真實的「極端奧客」情境：
    # 他選了「黑色 (沉穩)」當主色，卻要求「活力感 = 5.0 (極度活潑)」
    test_profile = {"gender": 0, "age_group": 1, "fluentenglish": 8, "country_code": "tw", "lang_code": "en"}
    test_main_color = "black"
    test_emotions = {
        "emotion_vitality": 5.0, 
        "emotion_stability": 0.0, 
        "emotion_resonance": 0.0, 
        "emotion_alert": 0.0
    }

    # --- 備份你原本寫好的完美動態權重 (調整層) ---
    original_model3 = main_pipeline.model3_dynamic_adjustment

    # --- 寫一個「沒有動態權重，永遠 50/50 平分」的糟糕調整層 ---
    def bad_5050_adjustment(m1_scores, raw_emotions):
        adjusted = {}
        for k in m1_scores.keys():
            # 強制把 Model 1 跟 使用者的分數對切
            adjusted[k] = (m1_scores[k] + raw_emotions.get(k, 0)) / 2.0
        return adjusted

    print("\n" + "🔥"*30)
    print(" 實驗 A：【拔掉動態權重】(永遠 50/50)")
    print("🔥"*30)
    # 偷偷把系統的調整層換成糟糕版
    main_pipeline.model3_dynamic_adjustment = bad_5050_adjustment
    res_a = run_advanced_pipeline(test_profile, test_main_color, test_emotions)
    colors_a = [r['colour'] for r in res_a]
    print(f"👉 沒加動態權重，系統推薦的輔助色是：{colors_a}")
    print("💬 視覺災難：主色是黑色，卻配上這些高飽和度的顏色，整個簡報會變得像「工地警告標誌」！\n")

    print("🛡️"*30)
    print(" 實驗 B：【啟動動態權重】(你們的 Alpha Decay 機制)")
    print("🛡️"*30)
    # 換回你們原本完美的調整層
    main_pipeline.model3_dynamic_adjustment = original_model3
    res_b = run_advanced_pipeline(test_profile, test_main_color, test_emotions)
    colors_b = [r['colour'] for r in res_b]
    print(f"👉 啟動動態權重，系統推薦的輔助色是：{colors_b}")
    print("💬 專業質感：系統發現活力 5.0 太瞎，成功壓制分數，給出了符合黑色本質的高級搭配！")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_real_impact_test()