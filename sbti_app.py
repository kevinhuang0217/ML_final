import streamlit as st
import numpy as np
import os

# ==========================================
# 1. 這裡貼上你們算出來的真實 Model 1 矩陣！
# (我先放模擬值，請務必用你們算出的真實數據覆蓋這裡)
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

# 系統會自動幫你算好大數據基準線！(不用自己算)
BIG_DATA_MEANS = np.mean(list(MODEL1_COLOR_MATRIX.values()), axis=0)

# ==========================================
# 2. 題目區 (特徵互斥設計：確保 16 種結果都會出現)
# ==========================================
sbti_questions = [
    {
        "id": "q1",
        "question": "Q1: 買高階耳機時，你會選哪種顏色？ (測試：高張揚 vs 高內斂)",
        "options": {
            "招搖熱血紅 (拉高活力)": {"red": 10},
            "寧靜高級灰 (拉高穩定)": {"grey": 10},
            "甜膩柔和粉 (拉高共鳴)": {"pink": 10},
            "科技冷感青 (拉高警示)": {"turquoise": 10}
        }
    },
    {
        "id": "q2",
        "question": "Q2: 民宿療癒空間，你最想走進哪一個？ (測試：群體取向 vs 獨立空間)",
        "options": {
            "無邊際星空露台 (黑+藍：穩定與警示)": {"black": 5, "blue": 5},
            "熱帶珊瑚礁 (紅+橘：極限活力)": {"red": 5, "orange": 5},
            "沉靜小木屋 (棕+綠：極限共鳴)": {"brown": 5, "green": 5},
            "極簡溫泉池 (白+灰：純粹穩定)": {"white": 5, "grey": 5}
        }
    },
    {
        "id": "q3",
        "question": "Q3: 夢見神獸，牠是什麼樣的生物？ (測試：溫暖能量 vs 冷靜距離)",
        "options": {
            "活潑柴犬 (黃+橘：溫暖活力)": {"yellow": 5, "orange": 5},
            "神秘九尾狐 (白+青：冷靜距離)": {"white": 5, "turquoise": 5},
            "古老巨龜 (綠+棕：溫和共鳴)": {"green": 5, "brown": 5},
            "華麗鳳凰 (紅+黑：強烈警示)": {"red": 5, "black": 5}
        }
    },
    {
        "id": "q4",
        "question": "Q4: 早八遲到衝出宿舍，抓哪件外套？ (測試：危機處理機制)",
        "options": {
            "隱蔽低調：純黑/鐵灰 (當透明人)": {"black": 5, "grey": 5},
            "極限衝刺：亮黃/鮮橘 (腎上腺素)": {"yellow": 5, "orange": 5},
            "裝可憐求情：奶茶棕/米白 (尋求同情)": {"brown": 5, "white": 5},
            "故作鎮定：海軍藍/紫色 (氣勢不能輸)": {"blue": 5, "purple": 5}
        }
    }
]

# ==========================================
# 3. 16 型人格資料庫 (Demo 版，請自行填入你們的文案)
# ==========================================
sbti_results = {
    "vsra": {"name": "快俠", "motto": "哈...哈...", "image": "assets/images/flash.png", "desc": "已經把「慢」昇華成防禦盾牌。"},
    "vsrA": {"name": "章魚哥", "motto": "我只在乎六點下班", "image": "assets/images/octupus.png", "desc": "看破資本主義的社畜。"},
    "vsRa": {"name": "統神", "motto": "我端火鍋摔倒，我一步都沒有退欸！", "image": "assets/images/godtone.png", "desc": "大腦內建無敵防禦機制。"},
    "vSra": {"name": "盧廣仲", "motto": "為什麼我會在台上", "image": "assets/images/crowdlu.png", "desc": "順水推舟氣死普通人。"},
    "Vsra": {"name": "阿公遛妻", "motto": "阿公67", "image": "assets/images/sixseven.png", "desc": "被網路迷因奪舍的行屍走肉。"},
    "vsRA": {"name": "周杰倫", "motto": "哎呦不錯喔", "image": "assets/images/jaychou.png", "desc": "裝逼界祖師爺。"},
    "vSrA": {"name": "崴寶", "motto": "草莓奶油裡面還有顆粒", "image": "assets/images/wei.png", "desc": "細節強迫症患者。"},
    "vSRa": {"name": "曾國城", "motto": "燒壞的還有三個人的未來", "image": "assets/images/chainsmoker.png", "desc": "通靈大師。"},
    "VsrA": {"name": "過動吉吉", "motto": "我只想好好學習", "image": "assets/images/adhd.png", "desc": "高敏感防禦暴躁吉娃娃。"},
    "VsRa": {"name": "孟寶", "motto": "有時候孟寶是怪怪寶", "image": "assets/images/meng.png", "desc": "談戀愛就化成糖水。"},
    "VSra": {"name": "芒果醬", "motto": "喔喔喔愛", "image": "assets/images/mangojump.png", "desc": "純愛戰士。"},
    "vSRA": {"name": "黃大謙", "motto": "我買了一個45美元的燈", "image": "assets/images/bigyellow.png", "desc": "厭世冷面笑匠。"},
    "VsRA": {"name": "阿志", "motto": "我想給天下的女孩一個家", "image": "assets/images/seventeen.png", "desc": "行動義氣發電機。"},
    "VSrA": {"name": "王ADEN", "motto": "把最後一段舞給同學", "image": "assets/images/aden.png", "desc": "主角光環妄想症。"},
    "VSRa": {"name": "香蕉葛葛", "motto": "我來到卡加布列島", "image": "assets/images/bananagg.png", "desc": "熱帶水果人生態度。"},
    "VSRA": {"name": "黑皮體育生", "motto": "理組有比較驕傲嗎", "image": "assets/images/happy.png", "desc": "把人生當大隊接力跑。"}
}

# ==========================================
# 4. Streamlit UI 
# ==========================================
def main():
    st.set_page_config(page_title="大數據心理測驗 Demo", page_icon="🎓")
    st.title("🎓 SBTI 數據驅動心理測驗 Demo")
    
    with st.form("sbti_form"):
        user_selections = []
        for q in sbti_questions:
            st.markdown(f"**{q['question']}**")
            choice = st.radio(label="請選擇：", options=list(q["options"].keys()), key=q["id"], label_visibility="collapsed")
            user_selections.append(q["options"][choice])
            st.write("") 
        submitted = st.form_submit_button("🔮 結算人格")

    if submitted:
        st.divider()
        # 計算及格線 (4題 x 10點 = 40點)
        threshold = BIG_DATA_MEANS * 40.0
        
        # 計算玩家總分
        user_total = np.zeros(4)
        for selection_dict in user_selections:
            for color, points in selection_dict.items():
                user_total += points * np.array(MODEL1_COLOR_MATRIX[color])
                
        # 大小寫判定
        v_letter = "V" if user_total[0] >= threshold[0] else "v"
        s_letter = "S" if user_total[1] >= threshold[1] else "s"
        r_letter = "R" if user_total[2] >= threshold[2] else "r"
        a_letter = "A" if user_total[3] >= threshold[3] else "a"
        final_code = f"{v_letter}{s_letter}{r_letter}{a_letter}"
        
        # 渲染結果
        st.success(f"🧬 你的大數據情緒編碼為：**{final_code}**")
        result_data = sbti_results.get(final_code, {"name": "神秘人格", "motto": "查無此人", "image": "", "desc": "演算法外星人"})
        
        st.markdown(f"### 🎉 真實人格：【{result_data['name']}】")
        st.caption(f"_{result_data['motto']}_")
        
        image_path = result_data.get("image", "")
        if image_path and os.path.exists(image_path):
            st.image(image_path, width=300)
        else:
            st.warning(f"⚠️ 提示：未讀取到圖片，請將圖片存放在 `{image_path}`")
            
        st.info(result_data["desc"])

if __name__ == "__main__":
    main()