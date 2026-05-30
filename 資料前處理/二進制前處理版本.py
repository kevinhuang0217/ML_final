import pandas as pd
import numpy as np

# ============================================================
# 載入資料
# ============================================================
file_path = 'preprocessing_DATA.csv'

df = pd.read_csv(file_path)
print(f"原始筆數: {len(df)}")
    
# ============================================================
# 步驟1：刪除不需要的欄位
# ============================================================
drop_cols = [
    'user',           # 識別碼
    'start', 'end',   # 填寫時間
    'lang', 'lang_full',      # 語言（已有mothertongue）
    'origincountry', 'origincountry_full',  # 出生國（保留居住國）
    'troubleseeing',  # 色覺問題
    'colorimportance' # 對顏色重視程度
]
df = df.drop(columns=drop_cols)

# ============================================================
# 步驟2：用 total_time 與 time_first4 篩選異常填寫時間
# ============================================================
before = len(df)

# 刪除缺失時間的資料
df = df.dropna(subset=['total_time', 'time_first4'])

# total_time：保留 120 ~ 1536 秒
df = df[(df['total_time'] >= 120) & (df['total_time'] <= 1536)]

# time_first4：保留 16 ~ 211 秒
df = df[(df['time_first4'] >= 16) & (df['time_first4'] <= 211)]

print(f"時間篩選後筆數: {len(df)}（刪除 {before - len(df)} 筆）")

# 篩選完後刪除時間欄位（不作為模型特徵）
df = df.drop(columns=['total_time', 'time_first4'])

# ============================================================
# 步驟3：處理缺失值
# ============================================================
before = len(df)

# 刪除 age / birthyear 缺失的資料
df = df.dropna(subset=['age'])

# 刪除情緒欄位有缺失的資料
emotion_cols = [
    'admiration', 'amusement', 'anger', 'compassion', 'contempt',
    'contentment', 'disappointment', 'disgust', 'fear', 'guilt',
    'hate', 'interest', 'joy', 'love', 'pleasure', 'pride',
    'regret', 'relief', 'sadness', 'shame'
]
df = df.dropna(subset=emotion_cols)

print(f"缺失值處理後筆數: {len(df)}（刪除 {before - len(df)} 筆）")

# ============================================================
# 步驟4：個人資訊欄位編碼
# ============================================================

# --- age：分組 ---
# 青年(18-35) / 中年(36-55) / 壯年(56+)
df = df.dropna(subset=['age'])
df['age'] = df['age'].astype(int)

def age_group(age):
    if age < 18:
        return 'minor'
    elif age <= 35:
        return 'young'
    elif age <= 55:
        return 'middle'
    else:
        return 'senior'

df['age_group'] = df['age'].apply(age_group)
df = df.drop(columns=['age', 'birthyear'])

# age_group 編碼
age_map = {'minor': 0, 'young': 1, 'middle': 2, 'senior': 3}
df['age_group'] = df['age_group'].map(age_map)

# --- gender 編碼 ---
# female=0, male=1, dnwta(不願透露)=2
gender_map = {'female': 0, 'male': 1, 'dnwta': 2}
df['gender'] = df['gender'].map(gender_map)

# ============================================================
# 工具函數：類別欄位 binary encoding
# ============================================================

def binary_encode_column(df, col, prefix, min_count=1000):
    """
    將類別欄位轉成 binary encoding。

    流程：
    1. 保留出現次數 >= min_count 的類別
    2. 其他類別改成 'other'
    3. 每個類別給一個整數編號
    4. 把整數編號轉成二進位欄位
    5. 回傳更新後的 df、對照表、產生的欄位名稱
    """

    # 1. 統計類別出現次數
    counts = df[col].value_counts()

    # 2. 找出要保留的類別
    keep_values = counts[counts >= min_count].index.tolist()

    # 3. 低於門檻者歸為 other
    df[col] = df[col].apply(
        lambda x: x if x in keep_values else 'other'
    )

    # 4. 建立類別清單
    categories = sorted(df[col].unique().tolist())

    # 讓 other 固定放最後，對照表比較好讀
    if 'other' in categories:
        categories = [x for x in categories if x != 'other'] + ['other']

    # 5. 建立類別 → 整數編號
    id_map = {
        category: idx
        for idx, category in enumerate(categories)
    }

    # 6. 把原本類別轉成整數 id
    encoded_id = df[col].map(id_map).astype(int)

    # 7. 計算需要幾個 bit
    n_categories = len(categories)
    n_bits = max(1, int(np.ceil(np.log2(n_categories))))

    # 8. 建立 binary 欄位
    bit_cols = []

    for bit in range(n_bits):
        bit_col = f'{prefix}_bin_{bit + 1}'
        shift = n_bits - bit - 1

        df[bit_col] = ((encoded_id // (2 ** shift)) % 2).astype(int)
        bit_cols.append(bit_col)

    # 9. 建立對照表
    mapping_df = pd.DataFrame({
        col: categories,
        f'{prefix}_id': range(len(categories)),
        f'{prefix}_binary': [
            format(i, f'0{n_bits}b')
            for i in range(len(categories))
        ]
    })  
    # 10. 刪除原本類別欄位
    df = df.drop(columns=[col])

    return df, mapping_df, bit_cols

    # 10. 刪除原本類別欄位
    df = df.drop(columns=[col])

    return df, mapping_df, bit_cols
# --- residencecountry：binary encoding ---
df, country_mapping, country_binary_cols = binary_encode_column(
    df=df,
    col='residencecountry',
    prefix='country',
    min_count=1000
)

# --- mothertongue：binary encoding ---
df, lang_mapping, lang_binary_cols = binary_encode_column(
    df=df,
    col='mothertongue',
    prefix='lang',
    min_count=1000
)

# --- fluentenglish：已是數值（0-10），保留不動 ---

# ============================================================
# 步驟5：情緒欄位合併成4大類
# ============================================================

# 活力感：joy, amusement, pleasure, interest, love
df['emotion_vitality'] = df[['joy', 'amusement', 'pleasure', 'interest', 'love']].mean(axis=1)

# 穩定感：admiration, pride, contentment, relief, compassion
df['emotion_stability'] = df[['admiration', 'pride', 'contentment', 'relief', 'compassion']].mean(axis=1)

# 共鳴感：sadness, disappointment, regret, guilt, shame
df['emotion_resonance'] = df[['sadness', 'disappointment', 'regret', 'guilt', 'shame']].mean(axis=1)

# 警示感：anger, contempt, disgust, fear, hate
df['emotion_alert'] = df[['anger', 'contempt', 'disgust', 'fear', 'hate']].mean(axis=1)

# 刪除原始18個情緒欄位
df = df.drop(columns=emotion_cols)
# ============================================================
# 步驟6：colour 欄位編碼
# ============================================================
colour_map = {
    'black': 0, 'blue': 1, 'brown': 2, 'green': 3,
    'grey': 4, 'orange': 5, 'pink': 6, 'purple': 7,
    'red': 8, 'turquoise': 9, 'white': 10, 'yellow': 11
}
df['colour_code'] = df['colour'].map(colour_map)

# ============================================================
# 步驟7：輸出結果
# ============================================================
print(f"\n最終筆數: {len(df)}")
print(f"最終欄位數: {len(df.columns)}")
print(f"\n欄位列表:")
print(df.columns.tolist())

print(f"\n各顏色筆數:")
print(df['colour'].value_counts())

# 輸出 binary encoding 對照表
country_mapping.to_csv('country_binary_mapping.csv', index=False)
lang_mapping.to_csv('lang_binary_mapping.csv', index=False)

print("\n國家 binary encoding 對照表:")
print(country_mapping)

print("\n母語 binary encoding 對照表:")
print(lang_mapping)

print("\nCountry binary columns:")
print(country_binary_cols)

print("\nLanguage binary columns:")
print(lang_binary_cols)
df.to_csv('DATA_preprocessed_try.csv', index=False)
print("\n已儲存 DATA_preprocessed.csv")