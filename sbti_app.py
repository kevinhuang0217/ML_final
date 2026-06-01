import streamlit as st
import numpy as np
import itertools

# ==========================================
# 0. 魔法換膚術 (注入朋友的美編 CSS)
# ==========================================
def inject_custom_css():
    st.markdown("""
    <style>
    /* 全局字體與漸層背景 */
    .stApp {
        background: linear-gradient(135deg, rgba(242, 201, 76, 0.22), rgba(19, 166, 161, 0.18) 36%, rgba(232, 93, 158, 0.18) 70%, rgba(255, 253, 248, 0.88)), #fff7e8;
        font-family: "Microsoft JhengHei", "Noto Sans TC", "PingFang TC", Arial, sans-serif;
        color: #171717;
    }
    
    /* 隱藏預設的頂部裝飾條 */
    header {visibility: hidden;}

    /* 題目表單外框 (玻璃透視感卡片) */
    [data-testid="stForm"] {
        background: rgba(255, 253, 248, 0.94);
        border: 1px solid rgba(23, 23, 23, 0.14);
        border-radius: 8px;
        box-shadow: 0 18px 60px rgba(15, 15, 15, 0.18);
        padding: 2rem;
    }

    /* 送出按鈕美化 (極簡純黑) */
    [data-testid="stFormSubmitButton"] > button {
        background-color: #171717;
        color: #ffffff;
        border-radius: 8px;
        border: 1px solid #171717;
        padding: 10px 16px;
        font-weight: 900;
        transition: all 0.2s ease;
        width: 100%;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        background-color: #333333;
        color: #ffffff;
        transform: translateY(-2px);
    }

    /* 單選框 (Radio選項) 轉化為卡片按鈕 */
    div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.66);
        border: 1px solid rgba(23, 23, 23, 0.14);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        transition: transform 150ms ease, border-color 150ms ease, background 150ms ease, box-shadow 150ms ease;
        cursor: pointer;
    }
    div[role="radiogroup"] > label:hover {
        transform: translateY(-1px);
        background: #fff;
        border-color: rgba(23, 23, 23, 0.32);
        box-shadow: 0 10px 28px rgba(23, 23, 23, 0.08);
    }

    /* 測驗結果區塊專屬漸層 */
    [data-testid="stNotification"] {
        background: linear-gradient(135deg, rgba(217, 54, 62, 0.12), rgba(47, 143, 91, 0.12) 35%, rgba(19, 166, 161, 0.12) 68%, rgba(232, 93, 158, 0.12)), rgba(255, 255, 255, 0.62);
        border: 1px solid rgba(23, 23, 23, 0.14);
        border-radius: 8px;
        color: #171717;
    }
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# 1. 核心大數據矩陣 (Model 1 權重)
# ==========================================
MODEL1_COLOR_MATRIX = {
    "black": [1.2, 4.8, 1.5, 4.2], "blue": [1.5, 4.5, 4.0, 1.1],
    "brown": [1.4, 3.9, 4.0, 2.4], "green": [2.2, 4.2, 4.5, 0.9],
    "grey": [1.1, 4.0, 1.2, 3.0], "orange": [4.3, 1.6, 3.1, 2.8],
    "pink": [3.5, 2.1, 4.5, 1.5], "purple": [2.8, 2.5, 3.9, 3.1],
    "red": [4.8, 1.2, 3.0, 4.5], "turquoise": [2.5, 3.5, 3.6, 1.3],
    "white": [2.0, 4.1, 2.9, 0.8], "yellow": [4.5, 1.9, 3.2, 2.5]
}

# ==========================================
# 2. 心理測驗題目區 (特徵解耦架構)
# ==========================================
sbti_questions = [
    {
        "id": "q1", 
        "question": "Q1: 早 9 的實驗課，遲到 15 分鐘學期總平均直接扣 20 分。你在早上 8:30 壓掉了鬧鐘，以為自己游刃有餘，結果玩了一把「再睡 5 分鐘」的危險小遊戲賭輸了... 再次睜開眼已經是 9:10。在這面臨死線、準備衝出宿舍的崩潰瞬間，你會隨手抓什麼顏色的衣服套在身上？",
        "options": {
            "隱蔽低調的純黑與深灰 (只想當個沒有靈魂的透明人)": {"black": 3, "grey": 3, "blue": 1, "white": 1},
            "腎上腺素爆發的亮黃與鮮橘 (準備極限百米衝刺)": {"yellow": 3, "orange": 3, "red": 1, "turquoise": 1},
            "裝可憐求同情的奶茶棕與米白 (喚醒助教惻隱之心)": {"brown": 3, "white": 3, "pink": 1, "green": 1},
            "故作鎮定的海軍藍與俐落白 (輸了成績但氣勢不能輸)": {"blue": 3, "white": 3, "black": 1, "grey": 1}
        }
    },
    {
        "id": "q2", 
        "question": "Q2: 中午 12 點，早上的最後一堂課終於結束。正午的陽光有些刺眼，當螢幕亮起、還沒解鎖的那一瞬間，映入眼簾的鎖定螢幕底色，是哪一種主色調？",
        "options": {
            "黑色 (Black)": {"black": 8}, "藍色 (Blue)": {"blue": 8},
            "棕色 (Brown)": {"brown": 8}, "灰色 (Grey)": {"grey": 8},
            "白色 (White)": {"white": 8}, "青色 (Turquoise)": {"turquoise": 8},
            "紅色 (Red)": {"red": 8}, "橘色 (Orange)": {"orange": 8},
            "黃色 (Yellow)": {"yellow": 8}, "綠色 (Green)": {"green": 8},
            "粉色 (Pink)": {"pink": 8}, "紫色 (Purple)": {"purple": 8}
        }
    },
    {
        "id": "q3", 
        "question": "Q3: 凌晨三點，你盯著遲遲解不開的問題，決定把筆電闔上。在這一片漆黑之中，如果大腦會自動召喚出一種「顏色」來包覆你，進行「系統重置」，直覺映入眼簾的是？",
        "options": {
            "黑色 (Black)": {"black": 8}, "藍色 (Blue)": {"blue": 8},
            "棕色 (Brown)": {"brown": 8}, "灰色 (Grey)": {"grey": 8},
            "白色 (White)": {"white": 8}, "青色 (Turquoise)": {"turquoise": 8},
            "紅色 (Red)": {"red": 8}, "橘色 (Orange)": {"orange": 8},
            "黃色 (Yellow)": {"yellow": 8}, "綠色 (Green)": {"green": 8},
            "粉色 (Pink)": {"pink": 8}, "紫色 (Purple)": {"purple": 8}
        }
    },
    {
        "id": "q4", 
        "question": "Q4: 連續幾週的期中考讓你電力耗盡。你抵達神秘民宿，管家提供六個專屬療癒空間。憑直覺，身心俱疲的你最想走進哪一個？",
        "options": {
            "沉靜的林間小木屋 (聞得到木頭香，窗外有鳥鳴)": {"brown": 3, "green": 3, "grey": 1, "white": 1},
            "無邊際的星空露台 (仰望深邃夜空)": {"black": 3, "blue": 3, "purple": 1, "turquoise": 1},
            "充滿霧氣的極簡溫泉池 (灰白清水模，安靜得只剩水聲)": {"grey": 3, "white": 3, "black": 1, "blue": 1},
            "暖烘烘的午後玻璃溫室 (陽光灑落，桌上有柑橘與熱茶)": {"orange": 3, "yellow": 3, "brown": 1, "green": 1},
            "魔幻霓虹的微醺地下酒吧 (爵士樂與夢幻調酒)": {"pink": 3, "purple": 3, "red": 1, "orange": 1},
            "充滿生命力的熱帶珊瑚礁潛水 (色彩斑斕的魚群與珊瑚)": {"red": 3, "turquoise": 3, "pink": 1, "yellow": 1}
        }
    },
    {
        "id": "q5", 
        "question": "Q5: 晚上回家遛狗，你會幫柴犬選擇什麼顏色的牽繩組合？",
        "options": {
            "藍 ＋ 黃 (經典高對比，充滿活力卻不失沉穩)": {"blue": 3, "yellow": 3, "turquoise": 1, "orange": 1},
            "黑 ＋ 紅 (強烈個性組合，代表極致專注與牽絆)": {"black": 3, "red": 3, "grey": 1, "purple": 1},
            "綠 ＋ 青 (初夏森林組合，溫柔療癒的大自然氣息)": {"green": 3, "turquoise": 3, "blue": 1, "white": 1},
            "粉 ＋ 白 (棉花糖軟萌組合，充滿愛與純潔感)": {"pink": 3, "white": 3, "red": 1, "yellow": 1},
            "棕 ＋ 橘 (大地秋意組合，溫暖復古像拿鐵與柑橘)": {"brown": 3, "orange": 3, "green": 1, "yellow": 1},
            "紫 ＋ 灰 (神祕極簡組合，低調沉靜帶有魔幻感)": {"purple": 3, "grey": 3, "black": 1, "pink": 1}
        }
    }
]

# ==========================================
# 3. 16 型人格資料庫
# ==========================================
sbti_results = {
    "HHHH": {"name": "黑皮體育生 (HAPPY)", "motto": "真的覺得憑甚麼歧視文組？理組有比較驕傲嗎，有載具嗎", "image": "assets/images/happy.png", "desc": "這是一個精神分裂的究極完全體。內心小劇場瘋狂運轉著「憑甚麼歧視」，甚至連日常對話都能讓他燃起熊熊怒火。但最可怕的是，他的臉上居然能無縫切換出陽光笑容說「冰美式好囉！」這種把殺氣壓縮進服務業精神的偽裝術，絕對是演化奇蹟。"},
    "HHHL": {"name": "香蕉葛葛 (BananaGG)", "motto": "我來到一個島 它叫卡加布列島", "image": "assets/images/bananagg.png", "desc": "大腦皮層大概有一半是砂糖和粉紅泡泡。別人的世界是絕地求生，他的世界是歡樂新手村。他不是裝瘋賣傻，是真的覺得只要跟小黑猩猩和蜜蜂女皇跳舞，世界就能被拯救。溫暖到讓人毛骨悚然，正向到逼人發瘋。"},
    "HHLH": {"name": "王ADEN (ADEN)", "motto": "我應該要把該給同學的最後一段舞給同學", "image": "assets/images/aden.png", "desc": "嚴重的主角光環妄想症。滿腦子想在最後一首歌「大跳」，遇到同學衝上台，情緒會瞬間原地爆炸。但就算氣到想報警，最後的結論依然是把舞跳完，這份至死不渝的偶像包袱感天動地。"},
    "HHLL": {"name": "芒果醬 (MangoJump)", "motto": "喔喔喔愛 有你的將來 我對你的感情我講不出來", "image": "assets/images/mangojump.png", "desc": "把現實生活當廉價熱血電影在演的無腦浪漫派。金錢地位都是狗屁，只要有「喔喔喔愛」就能靠光合作用活下去。口袋只剩一百塊依然能在紅綠燈前深情款款。純度太高的純愛戰士。"},
    "HLHH": {"name": "阿志 (seventeen)", "motto": "我不渣 我只是想給天下的女孩都有一個家", "image": "assets/images/seventeen.png", "desc": "行動的義氣發電機。搖起「花手」轉速快到可以發電。把「一聲兄弟，一生兄弟」當最高指導原則，看似無法無天的社會邊緣人，最大的宏願居然是給女孩一個家。"},
    "HLHL": {"name": "孟寶 (Meng)", "motto": "有時候 孟寶是怪怪寶 還好 有崴寶細心保護", "image": "assets/images/meng.png", "desc": "一談戀愛就融化成糖水的極度依戀者。世界充滿紀念日與粉紅泡泡，雖然偶爾化身無理取鬧的怪怪寶，但只要專屬避風港出現，就能立刻恢復成最甜膩黏人的可愛生物。"},
    "HLLH": {"name": "過動吉吉 (ADHD)", "motto": "那裡沒有書，沒有考卷，太嘈雜了，我只想好好學習", "image": "assets/images/adhd.png", "desc": "靈魂是一隻極度敏感且無法忍受噪音的暴躁吉娃娃。對沒有書、考卷的場所感到生理不適，高敏感防禦機制就是用讀書來隔絕這個吵鬧的世界。"},
    "HLLL": {"name": "阿公遛妻 (sixseven)", "motto": "欸six seven 阿公67", "image": "assets/images/sixseven.png", "desc": "脫離碳基生物思考範疇，被迷因奪舍的行屍走肉。大腦沒有邏輯過濾系統，隨時播放毫無意義的洗腦神曲，存在本身就是對人類智商的最大挑釁。"},
    "LHHH": {"name": "黃大謙 (BigYellow)", "motto": "我買了一個45美元的燈", "image": "assets/images/bigyellow.png", "desc": "把厭世當呼吸的冷面笑匠。不需要浮誇動作，只要用超然姿態就能把消費主義陷阱嘲諷得體無完膚。他的冷漠不是裝的，是真的覺得在座各位都很可笑。"},
    "LHHL": {"name": "曾國城 (Chainsmoker)", "motto": "一根菸，燒壞的不只是規矩，還有三個人的未來", "image": "assets/images/chainsmoker.png", "desc": "把胡說八道昇華成哲學的通靈大師。能把古蹟抽菸包裝成世紀大災難，聽他講話覺得智商被摩擦，但深沉嘴臉又讓人想問他大盤怎麼走。"},
    "LHLH": {"name": "崴寶 (Wei)", "motto": "這個草莓奶油裡面還有顆粒！他的顆粒好好吃喔", "image": "assets/images/wei.png", "desc": "細節控裡的重度強迫症。對食物有變態執著，把自己活成需要被捧在手心上的巨嬰，任何不夠精緻、沒有顆粒感的事物都入不了法眼。"},
    "LHLL": {"name": "盧廣仲 (CrowdLu)", "motto": "為什麼我會在台上 就是因為我買不到票 yeah", "image": "assets/images/crowdlu.png", "desc": "廢柴程度達到天人合一。沒有規劃、沒有野心，一切順水推舟。像落葉飄到終點線隨口說句 yeah，就這樣氣死一票每天努力到吐血的普通人。"},
    "LLHH": {"name": "周杰倫 (JayChou)", "motto": "哎呦不錯喔", "image": "assets/images/jaychou.png", "desc": "裝逼界祖師爺。詞彙量被封印，永遠只會嘴角微揚給出極度敷衍卻充滿霸氣的五個字。自帶「早就看穿一切但懶得解釋」的高級氣場。"},
    "LLHL": {"name": "統神 (GodTone)", "motto": "我端火鍋摔倒，我一步都沒有退欸，啊這樣算我輸喔？", "image": "assets/images/godtone.png", "desc": "嘴硬的終極權威，內建無敵防禦機制。就算端火鍋摔個四腳朝天，只要他不覺得自己跌倒，錯的絕對是那盆火鍋跟地心引力。"},
    "LLLH": {"name": "章魚哥 (Octupus)", "motto": "不管你要我做什麼工作，我都不會做的。我只在乎六點準時下班。", "image": "assets/images/octupus.png", "desc": "看破資本主義謊言的極致社畜。靈魂早被榨乾，人生最高指導原則就是準時下班，試圖攔住他的人都是殺父仇人。"},
    "LLLL": {"name": "快俠 (Flash)", "motto": "哈......哈......哈......哈......", "image": "assets/images/flash.png", "desc": "時間流速跟凡人不同次元。世界快轉到5G，他還在撥接上網。他不是在擺爛，他是真的已經把「慢」昇華成了一種堅不可摧的物理防禦盾牌。"}
}

# ==========================================
# 4. Streamlit UI
# ==========================================
def main():
    st.set_page_config(page_title="色彩 SBTI 心理測驗", page_icon="🎓", layout="centered")
    
    # 🔥 執行 CSS 注入
    inject_custom_css()

    st.title("🎓 色彩 SBTI 心理測驗")
    st.markdown("用五個很荒謬但很真實的校園瞬間，測出你的 **V/S/R/A** 高低組合，以及對應的 16 種人格。")
    st.write("")

    with st.form("sbti_form"):
        user_selections = {}
        for q in sbti_questions:
            st.markdown(f"**{q['question']}**")
            choice = st.radio(label="請選擇：", options=list(q["options"].keys()), key=q["id"], label_visibility="collapsed")
            user_selections[q["id"]] = q["options"][choice]
            st.write("") 
        submitted = st.form_submit_button("🔮 結算我的大數據人格")

    if submitted:
        st.divider()
        with st.spinner("📊 正在進行維度解耦運算與大數據比對..."):
            
            # --- 計算四個維度的獨立及格線 (Median) ---
            q1_v_scores = [sum(pts * MODEL1_COLOR_MATRIX[color][0] for color, pts in opt.items()) for opt in sbti_questions[0]["options"].values()]
            q4_s_scores = [sum(pts * MODEL1_COLOR_MATRIX[color][1] for color, pts in opt.items()) for opt in sbti_questions[3]["options"].values()]
            q5_r_scores = [sum(pts * MODEL1_COLOR_MATRIX[color][2] for color, pts in opt.items()) for opt in sbti_questions[4]["options"].values()]
            
            q2_a_scores = [sum(pts * MODEL1_COLOR_MATRIX[color][3] for color, pts in opt.items()) for opt in sbti_questions[1]["options"].values()]
            q3_a_scores = [sum(pts * MODEL1_COLOR_MATRIX[color][3] for color, pts in opt.items()) for opt in sbti_questions[2]["options"].values()]
            q23_a_scores = [a2 + a3 for a2, a3 in itertools.product(q2_a_scores, q3_a_scores)]
            
            median_v, median_s, median_r, median_a = np.median(q1_v_scores), np.median(q4_s_scores), np.median(q5_r_scores), np.median(q23_a_scores)

            # --- 計算玩家實際分數 ---
            user_v = sum(pts * MODEL1_COLOR_MATRIX[color][0] for color, pts in user_selections["q1"].items())
            user_s = sum(pts * MODEL1_COLOR_MATRIX[color][1] for color, pts in user_selections["q4"].items())
            user_r = sum(pts * MODEL1_COLOR_MATRIX[color][2] for color, pts in user_selections["q5"].items())
            user_a = sum(pts * MODEL1_COLOR_MATRIX[color][3] for color, pts in user_selections["q2"].items()) + \
                     sum(pts * MODEL1_COLOR_MATRIX[color][3] for color, pts in user_selections["q3"].items())

            # --- 判定大小寫 ---
            v_letter = "H" if user_v >= median_v else "L"
            s_letter = "H" if user_s >= median_s else "L"
            r_letter = "H" if user_r >= median_r else "L"
            a_letter = "H" if user_a >= median_a else "L"
            
            final_code = f"{v_letter}{s_letter}{r_letter}{a_letter}"
        
        # --- 渲染結果畫面 ---
        st.success(f"🧬 經過獨立特徵解析，你的情緒編碼為：**{final_code}**")
        result_data = sbti_results.get(final_code, {"name": "神秘未定義人格", "motto": "查無此人", "image": "", "desc": ""})
        
        st.markdown(f"## 🎉 真實人格：【{result_data['name']}】")
        st.caption(f"_{result_data['motto']}_")
        
        image_path = result_data.get("image", "")
        if image_path:
            # 確保 Streamlit 能正確載入本地圖片
            import os
            if os.path.exists(image_path):
                st.image(image_path, use_container_width=True)
            else:
                st.info(f"🖼️ 提示：請將圖片放至相對路徑 `{image_path}` 即可顯示")
                
        st.write(result_data["desc"])

if __name__ == "__main__":
    main()