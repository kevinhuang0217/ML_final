import streamlit as st
import numpy as np

# ==========================================
# 1. 核心大數據矩陣 (Model 1 權重)
# ==========================================
MODEL1_COLOR_MATRIX = {
    "black": [1.2, 4.8, 1.5, 4.2],
    "blue": [1.5, 4.5, 4.0, 1.1],
    "brown": [1.4, 3.9, 4.0, 2.4],
    "green": [2.2, 4.2, 4.5, 0.9],
    "grey": [1.1, 4.0, 1.2, 3.0],
    "orange": [4.3, 1.6, 3.1, 2.8],
    "pink": [3.5, 2.1, 4.5, 1.5],
    "purple": [2.8, 2.5, 3.9, 3.1],
    "red": [4.8, 1.2, 3.0, 4.5],
    "turquoise": [2.5, 3.5, 3.6, 1.3],
    "white": [2.0, 4.1, 2.9, 0.8],
    "yellow": [4.5, 1.9, 3.2, 2.5]
}

# ==========================================
# 2. 心理測驗題目區 (完整對接你的 5 題表格)
# ==========================================
sbti_questions = [
    {
        "id": "q1",
        "question": "Q1: 早 9 的實驗課，遲到 15 分鐘學期總平均直接扣 20 分。你在早上 8:30 壓掉了鬧鐘，以為自己游刃有餘，結果玩了一把「再睡 5 分鐘」的危險小遊戲賭輸了... 再次睜開眼已經是 9:10。在這面臨死線、準備衝出宿舍的崩潰瞬間，你會隨手抓什麼顏色的衣服套在身上？",
        "options": {
            "隱蔽低調的純黑與深灰 (只想當個沒有靈魂的透明人)": {"black": 3, "grey": 3, "blue": 1, "white": 1},
            "腎上腺素爆發的亮黃與鮮橘 (閃開讓專業的來！準備極限百米衝刺)": {"yellow": 3, "orange": 3, "red": 1, "turquoise": 1},
            "裝可憐求同情的奶茶棕與米白 (試圖營造柔弱感，喚醒助教惻隱之心)": {"brown": 3, "white": 3, "pink": 1, "green": 1},
            "故作鎮定的海軍藍與俐落白 (像大老闆從容走進教室，氣勢不能輸)": {"blue": 3, "white": 3, "black": 1, "grey": 1}
        }
    },
    {
        "id": "q2",
        "question": "Q2: 中午 12 點，早上的最後一堂課終於結束。你走出冷氣房，正午的陽光有些刺眼，你習慣性地從口袋掏出手機確認新訊息。當螢幕亮起、但還沒解鎖的那一瞬間，映入眼簾的鎖定螢幕底色，是哪一種主色調？",
        "options": {
            "黑色 (Black)": {"black": 8},
            "藍色 (Blue)": {"blue": 8},
            "棕色 (Brown)": {"brown": 8},
            "灰色 (Grey)": {"grey": 8},
            "白色 (White)": {"white": 8},
            "青色 (Turquoise)": {"turquoise": 8},
            "紅色 (Red)": {"red": 8},
            "橘色 (Orange)": {"orange": 8},
            "黃色 (Yellow)": {"yellow": 8},
            "綠色 (Green)": {"green": 8},
            "粉色 (Pink)": {"pink": 8},
            "紫色 (Purple)": {"purple": 8}
        }
    },
    {
        "id": "q3",
        "question": "Q3: 凌晨三點，你盯著螢幕上的遲遲解不開的問題，大腦幾乎要當機了，你決定把筆電闔上一分鐘往後癱在椅背上。在這片漆黑之中，如果大腦會自動召喚出一種「顏色」包覆你，進行最需要的「系統重置與狀態切換」，你直覺映入眼簾的是？",
        "options": {
            "黑色 (Black)": {"black": 8},
            "藍色 (Blue)": {"blue": 8},
            "棕色 (Brown)": {"brown": 8},
            "灰色 (Grey)": {"grey": 8},
            "白色 (White)": {"white": 8},
            "青色 (Turquoise)": {"turquoise": 8},
            "紅色 (Red)": {"red": 8},
            "橘色 (Orange)": {"orange": 8},
            "黃色 (Yellow)": {"yellow": 8},
            "綠色 (Green)": {"green": 8},
            "粉色 (Pink)": {"pink": 8},
            "紫色 (Purple)": {"purple": 8}
        }
    },
    {
        "id": "q4",
        "question": "Q4: 連續幾週的期中考與作業讓你電力完全耗盡。你獨自踏上一場「尋找平靜」的旅程，抵達神秘民宿時，管家提供六個風格迥異的專屬療癒空間。憑直覺，身心俱疲的你最想走進哪一個空間待上一整天？",
        "options": {
            "沉靜的林間小木屋 (聞得到木頭與乾草香，窗外傳來鳥鳴)": {"brown": 3, "green": 3, "grey": 1, "white": 1},
            "無邊際的星空露台 (躺在躺椅上，仰望深邃夜空)": {"black": 3, "blue": 3, "purple": 1, "turquoise": 1},
            "充滿霧氣的極簡溫泉池 (灰白清水模，安靜得只剩水聲)": {"grey": 3, "white": 3, "black": 1, "blue": 1},
            "暖烘烘的午後玻璃溫室 (陽光灑落，桌上擺著柑橘與熱茶)": {"orange": 3, "yellow": 3, "brown": 1, "green": 1},
            "魔幻霓虹的微醺地下酒吧 (播放爵士樂，特調夢幻調酒)": {"pink": 3, "purple": 3, "red": 1, "orange": 1},
            "充滿生命力的熱帶珊瑚礁潛水 (身邊全是色彩斑斕魚群與珊瑚)": {"red": 3, "turquoise": 3, "pink": 1, "yellow": 1}
        }
    },
    {
        "id": "q5",
        "question": "Q5: 晚上回家了，今天輪到你遛狗，你會幫你的柴犬選擇什麼顏色的牽繩組合？",
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
# 3. 16 型人格資料庫 (維持不變)
# ==========================================
sbti_results = {
    # ------ 高活力 (H 開頭) ------
    "HHHH": {"name": "黑皮體育生 (HAPPY)", "motto": "真的覺得憑甚麼歧視文組？理組有比較驕傲嗎，有載具嗎", "image": "", "desc": "這是一個精神分裂的究極完全體。內心小劇場瘋狂運轉著「憑甚麼歧視」，甚至連日常對話都能讓他燃起熊熊怒火。但最可怕的是，他的臉上居然能無縫切換出陽光笑容說「冰美式好囉！」這種把殺氣壓縮進服務業精神的偽裝術，絕對是演化奇蹟。"},
    "HHHL": {"name": "香蕉葛葛 (BananaGG)", "motto": "我來到一個島 它叫卡加布列島", "image": "", "desc": "大腦皮層大概有一半是砂糖和粉紅泡泡。別人的世界是絕地求生，他的世界是歡樂新手村。他不是裝瘋賣傻，是真的覺得只要跟小黑猩猩和蜜蜂女皇跳舞，世界就能被拯救。溫暖到讓人毛骨悚然，正向到逼人發瘋。"},
    "HHLH": {"name": "王ADEN (ADEN)", "motto": "我應該要把該給同學的最後一段舞給同學", "image": "", "desc": "嚴重的主角光環妄想症。滿腦子想在最後一首歌「大跳」，遇到同學衝上台，情緒會瞬間原地爆炸。但就算氣到想報警，最後的結論依然是把舞跳完，這份至死不渝的偶像包袱感天動地。"},
    "HHLL": {"name": "芒果醬 (MangoJump)", "motto": "喔喔喔愛 有你的將來 我對你的感情我講不出來", "image": "", "desc": "把現實生活當廉價熱血電影在演的無腦浪漫派。金錢地位都是狗屁，只要有「喔喔喔愛」就能靠光合作用活下去。口袋只剩一百塊依然能在紅綠燈前深情款款。純度太高的純愛戰士。"},
    "HLHH": {"name": "阿志 (seventeen)", "motto": "我不渣 我只是想給天下的女孩都有一個家", "image": "", "desc": "行動的義氣發電機。搖起「花手」轉速快到可以發電。把「一聲兄弟，一生兄弟」當最高指導原則，看似無法無天的社會邊緣人，最大的宏願居然是給女孩一個家。"},
    "HLHL": {"name": "孟寶 (Meng)", "motto": "有時候 孟寶是怪怪寶 還好 有崴寶細心保護", "image": "", "desc": "一談戀愛就融化成糖水的極度依戀者。世界充滿紀念日與粉紅泡泡，雖然偶爾化身無理取鬧的怪怪寶，但只要專屬避風港出現，就能立刻恢復成最甜膩黏人的可愛生物。"},
    "HLLH": {"name": "過動吉吉 (ADHD)", "motto": "那裡沒有書，沒有考卷，太嘈雜了，我只想好好學習", "image": "", "desc": "靈魂是一隻極度敏感且無法忍受噪音的暴躁吉娃娃。對沒有書、考卷的場所感到生理不適，高敏感防禦機制就是用讀書來隔絕這個吵鬧的世界。"},
    "HLLL": {"name": "阿公遛妻 (sixseven)", "motto": "欸six seven 阿公67", "image": "", "desc": "脫離碳基生物思考範疇，被迷因奪舍的行屍走肉。大腦沒有邏輯過濾系統，隨時播放毫無意義的洗腦神曲，存在本身就是對人類智商的最大挑釁。"},

    # ------ 低活力 (L 開頭) ------
    "LHHH": {"name": "黃大謙 (BigYellow)", "motto": "我買了一個45美元的燈", "image": "", "desc": "把厭世當呼吸的冷面笑匠。不需要浮誇動作，只要用超然姿態就能把消費主義陷阱嘲諷得體無完膚。他的冷漠不是裝的，是真的覺得在座各位都很可笑。"},
    "LHHL": {"name": "曾國城 (Chainsmoker)", "motto": "一根菸，燒壞的不只是規矩，還有三個人的未來", "image": "", "desc": "把胡說八道昇華成哲學的通靈大師。能把古蹟抽菸包裝成世紀大災難，聽他講話覺得智商被摩擦，但深沉嘴臉又讓人想問他大盤怎麼走。"},
    "LHLH": {"name": "崴寶 (Wei)", "motto": "這個草莓奶油裡面還有顆粒！他的顆粒好好吃喔", "image": "", "desc": "細節控裡的重度強迫症。對食物有變態執著，把自己活成需要被捧在手心上的巨嬰，任何不夠精緻、沒有顆粒感的事物都入不了法眼。"},
    "LHLL": {"name": "盧廣仲 (CrowdLu)", "motto": "為什麼我會在台上 就是因為我買不到票 yeah", "image": "", "desc": "廢柴程度達到天人合一。沒有規劃、沒有野心，一切順水推舟。像落葉飄到終點線隨口說句 yeah，就這樣氣死一票每天努力到吐血的普通人。"},
    "LLHH": {"name": "周杰倫 (JayChou)", "motto": "哎呦不錯喔", "image": "", "desc": "裝逼界祖師爺。詞彙量被封印，永遠只會嘴角微揚給出極度敷衍卻充滿霸氣的五個字。自帶「早就看穿一切但懶得解釋」的高級氣場。"},
    "LLHL": {"name": "統神 (GodTone)", "motto": "我端火鍋摔倒，我一步都沒有退欸，啊這樣算我輸喔？", "image": "", "desc": "嘴硬的終極權威，內建無敵防禦機制。就算端火鍋摔個四腳朝天，只要他不覺得自己跌倒，錯的絕對是那盆火鍋跟地心引力。"},
    "LLLH": {"name": "章魚哥 (Octupus)", "motto": "不管你要我做什麼工作，我都不會做的。我只在乎六點準時下班。", "image": "", "desc": "看破資本主義謊言的極致社畜。靈魂早被榨乾，人生最高指導原則就是準時下班，試圖攔住他的人都是殺父仇人。"},
    "LLLL": {"name": "快俠 (Flash)", "motto": "哈......哈......哈......哈......", "image": "", "desc": "時間流速跟凡人不同次元。世界快轉到5G，他還在撥接上網。他不是在擺爛，他是真的已經把「慢」昇華成了一種堅不可摧的物理防禦盾牌。"}
}

# ==========================================
# 4. Streamlit UI 與全局期望值計算引擎
# ==========================================
def main():
    st.set_page_config(page_title="校園 SBTI 心理測驗", page_icon="🎓", layout="centered")
    st.title("🎓 校園專屬 SBTI 數據驅動心理測驗")
    st.markdown("本測驗的 16 型人格切分標準，是由系統即時遍歷測驗組合、乘上 Model 1 權重後得出的**全局數學期望值 (Global Expected Value)**。")
    st.divider()

    with st.form("sbti_form"):
        user_selections = []
        for q in sbti_questions:
            st.markdown(f"**{q['question']}**")
            choice = st.radio(label="請選擇：", options=list(q["options"].keys()), key=q["id"], label_visibility="collapsed")
            user_selections.append(q["options"][choice])
            st.write("") 
        submitted = st.form_submit_button("🔮 結算我的大數據人格")

    if submitted:
        st.divider()
        st.subheader("📊 正在計算測驗全局期望值與大數據比對...")
        
        # 動態計算：所有可能組合的「完美期望值閾值」
        threshold = np.zeros(4) 
        
        for q in sbti_questions:
            q_options_scores = []
            for option_name, color_weights_dict in q["options"].items():
                option_vsra = np.zeros(4)
                for color, points in color_weights_dict.items():
                    option_vsra += points * np.array(MODEL1_COLOR_MATRIX[color])
                q_options_scores.append(option_vsra)
            threshold += np.mean(q_options_scores, axis=0)
            
        # 計算玩家總分
        user_total = np.zeros(4)
        for selection_dict in user_selections:
            for color, points in selection_dict.items():
                user_total += points * np.array(MODEL1_COLOR_MATRIX[color])
                
        # 大小寫判定 
        v_letter = "H" if user_total[0] >= threshold[0] else "L"
        s_letter = "H" if user_total[1] >= threshold[1] else "L"
        r_letter = "H" if user_total[2] >= threshold[2] else "L"
        a_letter = "H" if user_total[3] >= threshold[3] else "L"
        
        final_code = f"{v_letter}{s_letter}{r_letter}{a_letter}"
        
        # 渲染結果
        st.success(f"🧬 經過大數據比對，你的情緒編碼為：**{final_code}**")
        
        result_data = sbti_results.get(final_code, {
            "name": "神秘未定義人格", "motto": "查無此人", "image": "", "desc": "你超脫了三界之外。"
        })
        
        st.markdown(f"## 🎉 真實人格：【{result_data['name']}】")
        st.caption(f"_{result_data['motto']}_")
        
        image_path = result_data.get("image", "")
        if image_path:
            st.image(image_path, use_container_width=True)
        else:
            st.info("🖼️ 這裡未來會放迷因圖片喔！")
            
        st.write(result_data["desc"])

        st.markdown("""
        ---
        **【隱私與免責聲明】**
        純屬娛樂，不具專業心理學參考價值。我們不會收集你的任何資料，請安心服用！
        """)

if __name__ == "__main__":
    main()