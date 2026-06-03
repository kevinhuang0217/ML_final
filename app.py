import streamlit as st
import pandas as pd
import os
from main_pipeline import run_advanced_pipeline 

# ==========================================
# 1. 網頁基本設定與效能優化 (Caching)
# ==========================================
st.set_page_config(page_title="商業簡報視覺配色系統演算法", page_icon="🎨", layout="wide", initial_sidebar_state="expanded")

hide_streamlit_style = """
<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_data
def load_csv_mappings():
    cmap, lmap = None, None
    if os.path.exists("country_binary_mapping.csv"):
        cmap = pd.read_csv("country_binary_mapping.csv")
    if os.path.exists("lang_binary_mapping.csv"):
        lmap = pd.read_csv("lang_binary_mapping.csv")
    return cmap, lmap

country_mapping, lang_mapping = load_csv_mappings()

st.title("商業簡報視覺配色演算法")
st.markdown("##### 基於機器學習，為您的商業提案與專案報告計算最具說服力的視覺搭配。")
st.divider()

# ==========================================
# 2. 側邊欄 - 受眾輪廓設定
# ==========================================
# 🌍 擴充至 ISO 3166-1 標準的完整國家字典
COUNTRY_MAP = {
    "tw": "Taiwan", "cn": "China", "hk": "Hong Kong", "mo": "Macau", "jp": "Japan", "kr": "South Korea", 
    "sg": "Singapore", "my": "Malaysia", "id": "Indonesia", "th": "Thailand", "vn": "Vietnam", 
    "ph": "Philippines", "in": "India", "au": "Australia", "nz": "New Zealand", "us": "United States", 
    "ca": "Canada", "mx": "Mexico", "br": "Brazil", "ar": "Argentina", "cl": "Chile", "co": "Colombia", 
    "pe": "Peru", "gb": "United Kingdom", "uk": "United Kingdom", "ie": "Ireland", "fr": "France", 
    "de": "Germany", "es": "Spain", "it": "Italy", "pt": "Portugal", "nl": "Netherlands", "be": "Belgium", 
    "ch": "Switzerland", "at": "Austria", "se": "Sweden", "no": "Norway", "fi": "Finland", "dk": "Denmark", 
    "pl": "Poland", "ru": "Russia", "ua": "Ukraine", "gr": "Greece", "tr": "Turkey", "cz": "Czech Republic", 
    "ro": "Romania", "ae": "United Arab Emirates", "sa": "Saudi Arabia", "il": "Israel", "eg": "Egypt", 
    "za": "South Africa", "ng": "Nigeria", "ke": "Kenya", "hu": "Hungary", "bg": "Bulgaria", "hr": "Croatia", 
    "rs": "Serbia", "sk": "Slovakia", "si": "Slovenia", "ee": "Estonia", "lv": "Latvia", "lt": "Lithuania",
    "bd": "Bangladesh", "pk": "Pakistan", "lk": "Sri Lanka", "mm": "Myanmar (Burma)", "kh": "Cambodia",
    "ma": "Morocco", "dz": "Algeria", "tn": "Tunisia", "gh": "Ghana", "tz": "Tanzania"
}

# 🗣️ 擴充至 ISO 639-1 標準的完整語言字典
LANG_MAP = {
    "en": "English", "zh": "Chinese", "es": "Spanish", "fr": "French", "de": "German", "ja": "Japanese", 
    "ko": "Korean", "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ar": "Arabic", "hi": "Hindi",
    "nl": "Dutch", "pl": "Polish", "sv": "Swedish", "da": "Danish", "fi": "Finnish", "no": "Norwegian", 
    "el": "Greek", "he": "Hebrew", "ro": "Romanian", "hu": "Hungarian", "cs": "Czech", "uk": "Ukrainian",
    "ms": "Malay", "id": "Indonesian", "th": "Thai", "vi": "Vietnamese", "tl": "Tagalog", "bn": "Bengali", 
    "ta": "Tamil", "ur": "Urdu", "fa": "Persian", "tr": "Turkish", "sk": "Slovak", "bg": "Bulgarian", 
    "hr": "Croatian", "sr": "Serbian", "sl": "Slovenian", "et": "Estonian", "lv": "Latvian", "lt": "Lithuanian",
    "sw": "Swahili", "am": "Amharic", "yo": "Yoruba", "zu": "Zulu"
}

# 去除縮寫，只顯示國家/語言全名
def format_country(code):
    return COUNTRY_MAP.get(code.lower(), code.upper())

def format_lang(code):
    return LANG_MAP.get(code.lower(), code.upper())

with st.sidebar:
    st.header("👤 受眾設定")
    st.markdown("系統將依據您設定的受眾文化與背景，動態微調底層機率分佈。")
    
    gender_choice = st.radio("受眾主要性別？", ["女性", "男性", "不願透露"])
    age_choice = st.selectbox("受眾年齡層？", ["未成年 (<18)", "青年 (18-35)", "中年 (36-55)", "壯年 (56+)"], index=1)
    
    # 【防呆機制】：限制流利度區間 (6~9)，避免引發 OOD (Out-of-Distribution) 導致模型崩潰
    fluency_level = st.selectbox("受眾英語流利度", ["基礎 (6)", "中等 (7)", "流利 (8)", "精通 (9)"], index=2)
    fluent_val = int(fluency_level[-2]) # 自動擷取選項中的數字 (6, 7, 8, 9)
    
    st.markdown("---")
    st.markdown("#### 🌍 地緣與文化特徵")
    
    if country_mapping is not None:
        country_options = sorted(country_mapping['residencecountry'].dropna().unique().tolist())
        default_c = "us" if "us" in country_options else country_options[0]
        country_choice = st.selectbox("🌎 目標市場 (國家)", country_options, index=country_options.index(default_c), format_func=format_country)
    else:
        country_choice = st.text_input("🌎 國家代碼", value="us")
        
    if lang_mapping is not None:
        lang_options = sorted(lang_mapping['mothertongue'].dropna().unique().tolist())
        default_l = "en" if "en" in lang_options else lang_options[0]
        lang_choice = st.selectbox("🗣️ 溝通母語", lang_options, index=lang_options.index(default_l), format_func=format_lang)
    else:
        lang_choice = st.text_input("🗣️ 母語代碼", value="en")
    
    gender_map = {"女性": 0, "男性": 1, "不願透露": 2}
    age_map = {"未成年 (<18)": 0, "青年 (18-35)": 1, "中年 (36-55)": 2, "壯年 (56+)": 3}
    user_profile = {"gender": gender_map[gender_choice], "age_group": age_map[age_choice], "fluentenglish": fluent_val, "country_code": country_choice, "lang_code": lang_choice}

# ==========================================
# 3. 主畫面 - 商業級情緒微調介面
# ==========================================
all_colors = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'turquoise', 'white', 'yellow']

st.subheader("1️⃣ 定義品牌/簡報主色 (Primary Color)")
main_color = st.selectbox("請選擇您這份提案預設的「主要底色」：", all_colors)

st.divider()

st.subheader("2️⃣ 策略性視覺感受 (Strategic Visual Impact)")
st.markdown("請根據商業意圖調控以下情緒分數。這將引導演算法為您推薦最佳配色。")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 🏃 活力動能 (Vitality)")
    v = st.slider("活力感分數", 0.0, 5.0, 2.5, 0.1, label_visibility="collapsed")
    st.markdown("#### 💧 情感共鳴 (Resonance)")
    r = st.slider("共鳴感分數", 0.0, 5.0, 2.5, 0.1, label_visibility="collapsed")

with col2:
    st.markdown("#### 🧘 專業穩定 (Stability)")
    s = st.slider("穩定感分數", 0.0, 5.0, 2.5, 0.1, label_visibility="collapsed")
    st.markdown("#### ⚠️ 警示驅動 (Alert)")
    a = st.slider("警示感分數", 0.0, 5.0, 2.5, 0.1, label_visibility="collapsed")

user_emotions = {"emotion_vitality": v, "emotion_stability": s, "emotion_resonance": r, "emotion_alert": a}

st.divider()

# ==========================================
# 4. 色碼與 XAI 解釋字典
# ==========================================
UI_COLOR_MAP = {
    "black": "#1E293B", "blue": "#3B82F6", "brown": "#92400E", "green": "#10B981", 
    "grey": "#64748B", "orange": "#F97316", "pink": "#EC4899", "purple": "#8B5CF6", 
    "red": "#EF4444", "turquoise": "#06B6D4", "white": "#F8FAFC", "yellow": "#EAB308"
}

XAI_EXPLANATION = {
    "black": "深邃且具備絕對權威感，在色彩心理學中能最大化『穩定感』與高端商業價值，適合奠定不可動搖的專業基調。",
    "blue": "全球百大企業最愛的信任色。具備極高的『穩定』與『理性』特質，能有效降低受眾防備心並提升說服力。",
    "brown": "帶有大地般的沉穩與務實感，能建立溫暖且可靠的『共鳴感』，適合強調永續、傳統或穩健成長的提案。",
    "green": "象徵成長、和平與生機。能帶來極佳的『共鳴感』與『穩定感』，是推動 ESG、環保或友善創新的首選視覺。",
    "grey": "中立且不喧賓奪主的高級背景色。能完美襯托數據與重點，提供無壓力的閱讀體驗，展現極致的『專業穩定』。",
    "orange": "充滿熱情與平易近人的擴張色。能激發高度的『活力動能』，且比紅色更具親和力，適合促銷或激勵型提案。",
    "pink": "具備柔和與創新的顛覆性特質。能創造強烈的『情感共鳴』與記憶點，適合打破常規、強調溫柔堅定的破壞式創新。",
    "purple": "融合了紅色的活力與藍色的穩定，帶有神秘與尊貴感。適合強調『獨特性』、『奢華』或『前瞻性科技』的場景。",
    "red": "最具視覺衝擊力的高能色彩。具備頂尖的『警示驅動』與『活力動能』，能瞬間聚焦受眾視線並促成行動 (CTA)。",
    "turquoise": "結合藍色科技感與綠色生機。給人清晰、靈動的『活力感』，非常適合數位轉型、醫療科技或年輕品牌。",
    "white": "極簡、純粹且包容萬物。能提供最大的呼吸空間，讓內容本身成為主角，是現代極簡商業設計的終極武器。",
    "yellow": "自帶光源的極亮色。能爆發出最強的『活力動能』與樂觀情緒，適合激發創意、傳遞希望或作為強烈的重點提示。"
}

def get_text_color(bg_color_name):
    dark_colors = ['black', 'blue', 'brown', 'green', 'purple', 'red', 'grey']
    return "white" if bg_color_name.lower() in dark_colors else "#1E293B"

# ==========================================
# 5. 執行推論與結果展示
# ==========================================
if st.button("✨ 啟動 AI 演算法計算最佳配色", use_container_width=True, type="primary"):
    with st.spinner("AI 引擎運算中 (貝氏機率分佈與退火尋優)..."):
        
        recommendations = run_advanced_pipeline(user_profile, main_color, user_emotions)
        st.success("✅ 分析完成！")
        
        top3_raw_sum = sum([r['prob'] for r in recommendations])
        
        c1, c2, c3 = recommendations[0]['colour'], recommendations[1]['colour'], recommendations[2]['colour']
        hex_main = UI_COLOR_MAP.get(main_color, main_color)
        hex_c1, hex_c2, hex_c3 = UI_COLOR_MAP.get(c1, c1), UI_COLOR_MAP.get(c2, c2), UI_COLOR_MAP.get(c3, c3)
        
        st.subheader(f"🎨 針對主色「{main_color.upper()}」，系統推薦的最佳搭配：")
        
        res_cols = st.columns(3)
        for idx, col in enumerate(res_cols):
            if idx < len(recommendations):
                rec = recommendations[idx]
                real_prob = rec['prob']
                
                if real_prob >= 0.30: badge = "🔥 Excellent"
                elif real_prob >= 0.15: badge = "⭐ good"
                else: badge = "💡 considerable"
                    
                col.metric(
                    label=f"🏆 第 {idx+1} 名", 
                    value=rec['colour'].capitalize(), 
                    delta=f"機率: {real_prob:.1%} | {badge}",
                    delta_color="off"
                )
                
        # 實體配色預覽
        st.write("")
        st.markdown("#### 👀 實體配色預覽 (Palette Preview)")
        
        main_border = "border: 1px solid #E2E8F0;" if main_color == "white" else ""
        html_code = f"""
        <div style="display: flex; border-radius: 12px; overflow: hidden; height: 120px; box-shadow: 0 4px 10px rgba(0,0,0,0.15); margin-top: 10px;">
            <div style="{main_border} flex: 4; background-color: {hex_main}; color: {get_text_color(main_color)}; display: flex; align-items: center; justify-content: center; font-size: 22px; font-weight: bold; border-right: 2px solid rgba(255,255,255,0.3);">主色 {main_color.capitalize()}</div>
            <div style="flex: 2; background-color: {hex_c1}; color: {get_text_color(c1)}; display: flex; flex-direction: column; align-items: center; justify-content: center; font-weight: bold; font-size: 16px;">
                <span style="font-size: 12px; opacity: 0.8;">Top 1</span><span>{c1.capitalize()}</span>
            </div>
            <div style="flex: 2; background-color: {hex_c2}; color: {get_text_color(c2)}; display: flex; flex-direction: column; align-items: center; justify-content: center; font-weight: bold; font-size: 16px;">
                <span style="font-size: 12px; opacity: 0.8;">Top 2</span><span>{c2.capitalize()}</span>
            </div>
            <div style="flex: 2; background-color: {hex_c3}; color: {get_text_color(c3)}; display: flex; flex-direction: column; align-items: center; justify-content: center; font-weight: bold; font-size: 16px;">
                <span style="font-size: 12px; opacity: 0.8;">Top 3</span><span>{c3.capitalize()}</span>
            </div>
        </div>
        """
        st.markdown(html_code, unsafe_allow_html=True)
        st.write("")

        # 【魔王功能 2：一鍵複製色碼 (包含主色)】
        st.markdown("##### ✂️ 一鍵複製色碼 (Hex Codes)")
        code_main, code_col1, code_col2, code_col3 = st.columns(4)
        with code_main: 
            st.caption("主色 (Primary)")
            st.code(hex_main, language=None)
        with code_col1: 
            st.caption("Top 1")
            st.code(hex_c1, language=None)
        with code_col2: 
            st.caption("Top 2")
            st.code(hex_c2, language=None)
        with code_col3: 
            st.caption("Top 3")
            st.code(hex_c3, language=None)

        # 【魔王功能 3：XAI 可解釋性】
        st.write("")
        with st.expander("💡 為什麼 AI 推薦這些顏色？ (點擊查看模型可解釋性 XAI)"):
            st.markdown(f"**🥇 Top 1 ({c1.capitalize()}):** {XAI_EXPLANATION.get(c1, '')}")
            st.markdown(f"**🥈 Top 2 ({c2.capitalize()}):** {XAI_EXPLANATION.get(c2, '')}")
            st.markdown(f"**🥉 Top 3 ({c3.capitalize()}):** {XAI_EXPLANATION.get(c3, '')}")