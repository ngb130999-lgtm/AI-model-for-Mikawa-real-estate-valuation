import pandas as pd
import numpy as np
import glob
import os
import re
from category_encoders import TargetEncoder
import joblib
from sklearn.cluster import KMeans

# --- 1. データクリーニング関数 ---

def clean_station_distance(time_val):
    """最寄駅からの距離（徒歩分）を数値化する関数"""
    if pd.isna(time_val): return 120.0 
    if not isinstance(time_val, str): return float(time_val)
    # 全角数字を半角に変換し、不要な空白を除去
    time_val = (time_val.translate(str.maketrans('０１２３４５６７８９', '0123456789')).strip().replace(" ", ""))
    
    def parse_segment(segment):
        if not segment: return None
        if "H" in segment: # 時間表記の処理
            match = re.search(r"(\d+)H(\d+)?", segment)
            if match:
                h = int(match.group(1))
                m = int(match.group(2)) if match.group(2) else 0
                return h * 60 + m
        if "分" in segment: # 分表記の処理
            match = re.search(r"(\d+)分", segment)
            if match: return int(match.group(1))
        if segment.isdigit(): return int(segment)
        return None

    if "～" in time_val: # 範囲表記（例：10分～15分）の中間値を取得
        parts = time_val.split("～")
        min_p = parse_segment(parts[0]); max_p = parse_segment(parts[1])
        return (min_p + max_p) / 2 if (min_p and max_p) else (min_p or max_p)
    return parse_segment(time_val)

def clean_area(value):
    """面積（㎡）の文字列を数値に変換する関数"""
    if pd.isna(value): return None
    s = str(value).replace("㎡以上", "").replace(",", "")
    return pd.to_numeric(s, errors='coerce')

# --- 2. データの読み込みと一次前処理 ---

def load_data(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    df = pd.concat([pd.read_csv(f, encoding="cp932") for f in all_files], ignore_index=True)
    
    # 対象を「宅地(土地と建物)」に限定
    df = df[df["種類"] == "宅地(土地と建物)"].copy()
    
    # 位置情報の統合キー作成（市区町村 + 地区 + 駅名）
    df["LocationKey"] = (df["市区町村名"].fillna("Unk") + "_" + 
                         df["地区名"].fillna("Unk") + "_" + 
                         df["最寄駅：名称"].fillna("Unk"))

    column_map = {
        "市区町村名": "City",
        "取引価格（総額）": "Price",
        "建物の構造": "Structure",
        "前面道路：幅員（ｍ）": "Road_Width",
        "都市計画": "City_Planning"
    }
    df = df.rename(columns=column_map)
    
    # 取引時期から年を抽出
    df["Year"] = df["取引時期"].str.extract(r"(\d{4})").astype(float)
    df = df.dropna(subset=['Year', 'Price'])
    
    # 築年数の計算
    df["Year_Built"] = pd.to_numeric(df["建築年"].str.replace("年", "", regex=False), errors="coerce")
    df["Building_Age"] = df["Year"] - df["Year_Built"]
    df["Building_Age"] = df["Building_Age"].apply(lambda x: x if x >= 0 else 0)

    # クリーニング関数の適用
    df["Station_Distance"] = df["最寄駅：距離（分）"].apply(clean_station_distance)
    df["Area_m2"] = df["面積（㎡）"].apply(clean_area)
    df["Floor_Area_m2"] = df["延床面積（㎡）"].apply(clean_area)

    # --- [手法 2: ターゲット変数の修正] ---
    # 総額の代わりに、㎡単価（Price_per_m2）を予測対象とする
    # 敷地面積 0 以下のデータを除外
    df = df[df['Area_m2'] > 0].copy()
    df['Price_per_m2'] = df['Price'] / df['Area_m2']
    # ㎡単価を対数変換（正規分布化）
    df['Log_Price_per_m2'] = np.log1p(df['Price_per_m2'])

    # 特徴量エンジニアリング
    df["Efficiency_ratio"] = df["Floor_Area_m2"] / df["Area_m2"] # 容積率の実績
    df['Is_New_House'] = (df['Building_Age'] <= 5).astype(int)   # 築浅フラグ
    df['Is_Old_House'] = (df['Building_Age'] >= 22).astype(int)  # 築古フラグ
    df['Road_Land_Interaction'] = df['Road_Width'].fillna(0) * df['Area_m2'] # 接道状況と面積の積
    df['Is_Near_Station'] = (df['Station_Distance'] <= 10).astype(int)       # 駅近フラグ
    df['Log_Area'] = np.log1p(df['Area_m2']) # 面積の対数変換
    
    # 駅からの距離に非線形な変化（反比例）を導入
    df['Station_Dist_Inv'] = 1 / (df['Station_Distance'] + 1)
    
    return df

# --- 3. パイプラインの実行 ---

folder_path = "/home/giabao_1309/property/mikawa_property_dataset2015-2025"
df_raw = load_data(folder_path)

# 外れ値の除外（300万〜1.5億円、2000㎡未満に限定）
df_raw = df_raw[(df_raw['Price'] > 3_000_000) & (df_raw['Price'] < 150_000_000)]
df_raw = df_raw[df_raw['Area_m2'] < 2000]

# --- [手法 1: K-MEANS クラスタリング] ---
# 経済的立地（駅からの距離と㎡単価）に基づくエリアのクラスタリング
print("📌 エリアクラスタリング（K-Means）を実行中...")
clustering_features = df_raw[['Station_Distance', 'Log_Price_per_m2']].fillna(0)
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
df_raw['Area_Cluster'] = kmeans.fit_predict(clustering_features).astype(str)

# 時系列による学習データとテストデータの分割（2024年を境に分割）
train_df = df_raw[df_raw['Year'] < 2024].copy()
test_df = df_raw[df_raw['Year'] >= 2024].copy()

# 欠損値の補完（容積率に基づき延床面積を推計、その他は中央値で補完）
for d in [train_df, test_df]:
    mask = d["Floor_Area_m2"].isna()
    d.loc[mask, "Floor_Area_m2"] = (d.loc[mask, "Area_m2"] * d.loc[mask, "容積率（％）"].fillna(100) / 100)
    d["Road_Width"] = d["Road_Width"].fillna(train_df["Road_Width"].median())
    d["Building_Age"] = d["Building_Age"].fillna(train_df["Building_Age"].median())

# --- 4. ターゲットエンコーディング ---

# 'Area_Cluster' をカテゴリカル変数リストに追加
categorical_features = ["LocationKey", "City", "Structure", "City_Planning", "Area_Cluster"]
numeric_features = ["Station_Distance", "Area_m2", "Floor_Area_m2", "Building_Age", "Road_Width",
                    "Efficiency_ratio", "Is_New_House", "Is_Old_House", "Road_Land_Interaction", 
                    "Log_Area", "Is_Near_Station", "Station_Dist_Inv"]

te = TargetEncoder(cols=categorical_features)

# 新たな目標変数（㎡単価の対数）に対して学習を実行
train_encoded = te.fit_transform(train_df[categorical_features], train_df['Log_Price_per_m2'])
test_encoded = te.transform(test_df[categorical_features])

# 特徴量セットの結合
X_train = pd.concat([train_encoded, train_df[numeric_features]], axis=1)
X_test = pd.concat([test_encoded, test_df[numeric_features]], axis=1)

# 学習ターゲットの設定（㎡単価）
y_train = train_df['Log_Price_per_m2']
y_test = test_df['Log_Price_per_m2']

# --- 5. 結果の保存 ---
print(f"📊 完了! モデルは「㎡単価」を予測します。 X_train 形状: {X_train.shape}")

X_train.to_csv("mikawa_X_train_final.csv", index=False, encoding="utf-8-sig")
X_test.to_csv("mikawa_X_test_final.csv", index=False, encoding="utf-8-sig")
y_train.to_csv("mikawa_y_train.csv", index=False)
y_test.to_csv("mikawa_y_test.csv", index=False)

# モデル展開用にエンコーダーとクラスターモデルを保存
joblib.dump(te, "mikawa_target_encoder.joblib")
joblib.dump(kmeans, "mikawa_kmeans_cluster.joblib")