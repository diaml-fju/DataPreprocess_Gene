import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import NMF
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

# 設定頁面佈局
st.set_page_config(page_title="基因數據分析儀表板", layout="wide")

# ==========================================
# --- 核心運算函數區 ---
# ==========================================
def perform_clr_transformation(df_X):
    df_numeric = df_X.apply(pd.to_numeric, errors='coerce')
    n = len(df_numeric)
    impute_value = 1 / (n**2)
    impute_df = df_numeric.replace(0, impute_value).copy()
    df_prop = impute_df.div(impute_df.sum(axis=1), axis=0).copy()
    log_df = np.log(df_prop)
    clr_df = log_df.sub(log_df.mean(axis=1), axis=0)
    return clr_df, df_prop, impute_value

# (新增) 第四步函式 1：解析 H 矩陣
def extract_sorted_features_from_H(H_df, top_n=None):
    sorted_dict = {}
    for comp in H_df.index:
        comp_series = H_df.loc[comp]
        comp_sorted = comp_series.sort_values(ascending=False)
        if top_n is not None:
            comp_sorted = comp_sorted.head(top_n)
        
        comp_df = comp_sorted.reset_index()
        comp_df.columns = ["Feature", "Contribution"]
        total = comp_series.sum()
        comp_df["Percentage (%)"] = (comp_df["Contribution"] / total) * 100
        comp_df.insert(0, "Rank", range(1, len(comp_df) + 1))
        sorted_dict[f"{comp}"] = comp_df
    return sorted_dict

# (新增) 第四步函式 2：總結 W 矩陣
def summarize_W_components(W_df, class_col="Y"):
    df = W_df.copy()
    component_cols = df.columns.drop(class_col)
    df["Dominant_Component"] = df[component_cols].idxmax(axis=1)
    sample_summary = df
    group_mean = df.groupby(class_col)[component_cols].mean()
    dominant_counts = (
        df.groupby([class_col, "Dominant_Component"])
        .size()
        .unstack(fill_value=0)
    )
    return sample_summary, group_mean, dominant_counts

# (新增) 第四步函式 3：類別共用成分對比
def compare_components_by_class(W_df, class_col="Y"):
    df = W_df.copy()
    component_cols = df.columns.drop(class_col)
    sorted_components = df[component_cols].apply(
        lambda row: row.sort_values(ascending=False).index.tolist(),
        axis=1
    )
    sorted_per_sample = pd.DataFrame({
        "Class": df[class_col],
        "Sorted_Components": sorted_components
    })
    df["Dominant_Component"] = df[component_cols].idxmax(axis=1)
    class_component_sets = {}
    for cls in df[class_col].unique():
        comps = set(df[df[class_col] == cls]["Dominant_Component"])
        class_component_sets[cls] = comps
    
    classes = list(class_component_sets.keys())
    if len(classes) == 2:
        c0, c1 = classes
        shared = class_component_sets[c0] & class_component_sets[c1]
        unique_c0 = class_component_sets[c0] - class_component_sets[c1]
        unique_c1 = class_component_sets[c1] - class_component_sets[c0]
        comparison_summary = {
            f"Class {c0} components": class_component_sets[c0],
            f"Class {c1} components": class_component_sets[c1],
            "Shared components": shared,
            f"Unique to Class {c0}": unique_c0,
            f"Unique to Class {c1}": unique_c1
        }
    else:
        comparison_summary = class_component_sets
    return sorted_per_sample, class_component_sets, comparison_summary

# (新增) 第四步函式 4：Top-K 類別對比
def topk_components_class_comparison(W_df, class_col="Y", top_k=1):
    df = W_df.copy()
    component_cols = df.columns.drop(class_col)
    topk_components = df[component_cols].apply(
        lambda row: row.sort_values(ascending=False).index[:top_k].tolist(),
        axis=1
    )
    df["TopK_Components"] = topk_components
    topk_per_sample = df[[class_col, "TopK_Components"]]
    exploded = topk_per_sample.explode("TopK_Components")
    topk_counts = (
        exploded.groupby([class_col, "TopK_Components"])
        .size()
        .unstack(fill_value=0)
    )
    class_sets = {}
    for cls in df[class_col].unique():
        class_sets[cls] = set(topk_counts.loc[cls][topk_counts.loc[cls] > 0].index)
    
    classes = list(class_sets.keys())
    if len(classes) >= 2:
        c0, c1 = classes[:2]
        shared = class_sets[c0] & class_sets[c1]
        unique_c0 = class_sets[c0] - class_sets[c1]
        unique_c1 = class_sets[c1] - class_sets[c0]
        class_comparison = {
            f"Class {c0} Top-{top_k}": class_sets[c0],
            f"Class {c1} Top-{top_k}": class_sets[c1],
            "Shared components": shared,
            f"Unique to Class {c0}": unique_c0,
            f"Unique to Class {c1}": unique_c1
        }
    else:
        class_comparison = None
    return topk_per_sample, topk_counts, class_comparison

def compare_ranked_features_summary(
    df_list,
    df_names=None,
    feature_col=1,
    top_n=None,
    min_percentage=None,
    percentage_col=3):
    """
    Create a summary DataFrame showing feature presence across datasets.
    """
    
    if df_names is None:
        df_names = [f"DF_{i+1}" for i in range(len(df_list))]
    
    feature_sets = {}
    
    for name, df in zip(df_names, df_list):
        temp_df = df.copy()
        
        if top_n is not None:
            temp_df = temp_df.iloc[:top_n]
        
        if min_percentage is not None:
            temp_df = temp_df[temp_df.iloc[:, percentage_col] >= min_percentage]
        
        feature_sets[name] = set(temp_df.iloc[:, feature_col])
    
    # All unique features across datasets
    all_features = sorted(set.union(*feature_sets.values()))
    
    # Build summary table
    summary_df = pd.DataFrame(index=all_features)
    
    for name in df_names:
        summary_df[name] = summary_df.index.isin(feature_sets[name]).astype(int)
    
    # Add summary columns
    summary_df["Total_Appearance"] = summary_df[df_names].sum(axis=1)
    summary_df["Shared_in_All"] = (summary_df["Total_Appearance"] == len(df_names)).astype(int)
    
    return summary_df
# ==========================================
# --- 介面設計 ---
# ==========================================
st.title("🧬 Gene Study: 階段式自動化分析儀表板")

# 包含第四步的 Tab
tab1, tab2, tab3, tab4, tab5 = st.tabs(["第一步：CLR 前處理", "第二步：跨資料集對齊", "第三步：NMF 分解", "第四步：特徵解析與群組對比","第五步：機器學習"])

# ==========================================
# --- 第一步：CLR 前處理 ---
# ==========================================
with tab1:
    st.header("🛠️ 數據預處理階段")
    file_clr = st.file_uploader("📂 上傳 ASV Count CSV", type=["csv"], key="clr_upload")
    
    if file_clr:
        df_raw = pd.read_csv(file_clr)
        
        # --- Y 變數設定區 ---
        has_y_1 = st.checkbox("此資料集包含目標變數 (Y) 嗎？", value=True, key="hy1")
        if has_y_1:
            y_col_1 = st.selectbox("選擇目標變數 (Y) 欄位：", df_raw.columns, key="yc1")
            df_y = df_raw[[y_col_1]].copy()
            df_X = df_raw.drop(columns=[y_col_1])
        else:
            df_y = pd.DataFrame()
            df_X = df_raw.copy()

        st.subheader("1. 原始數據預覽")
        st.dataframe(df_raw.head(), use_container_width=True)

        with st.spinner('執行 CLR 運算中...'):
            clr_X, prop_X, imp_val = perform_clr_transformation(df_X)
        
        st.info(f"💡 樣本數 $n={len(df_raw)}$，自動補值：`{imp_val:.8f}`")

        # 縫合 Y 變數準備匯出
        if has_y_1:
            clr_final = pd.concat([df_y, clr_X], axis=1)
        else:
            clr_final = clr_X

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("2. CLR 轉換結果 (已縫合 Y)")
            st.dataframe(clr_final.head(), use_container_width=True)
        with col2:
            st.subheader("導出檔案")
            raw_name = os.path.splitext(file_clr.name)[0]
            st.download_button(
                label="📥 下載 CLR 結果",
                data=clr_final.to_csv(index=False).encode('utf-8'),
                file_name=f"{raw_name}_CLRTransformed.csv",
                mime='text/csv'
            )

        st.divider()
        st.subheader("3. 數據分佈對比 (KDE Plot)")
        sel_asvs = st.multiselect("選擇觀察特徵:", clr_X.columns.tolist(), clr_X.columns[:5].tolist())
        if sel_asvs:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
            sns.kdeplot(data=prop_X[sel_asvs], ax=ax1, fill=True, alpha=0.1)
            ax1.set_title("Original Proportions")
            sns.kdeplot(data=clr_X[sel_asvs], ax=ax2, fill=True, alpha=0.1)
            ax2.set_title("CLR Transformed")
            st.pyplot(fig)
    else:
        st.info("請上傳 CSV 檔案開始第一步處理。")


# ==========================================
# --- 第二步：跨資料集對齊與差異平移 ---
# ==========================================
with tab2:
    st.header("🔗 第二步：跨資料集對齊與差異平移")
    
    c1, c2 = st.columns(2)
    with c1:
        file_b = st.file_uploader("📂 上傳對照組 (Before) CSV", type=["csv"], key="b_up")
    with c2:
        file_a = st.file_uploader("📂 上傳實驗組 (After) CSV", type=["csv"], key="a_up")

    if file_b and file_a:
        df_b = pd.read_csv(file_b)
        df_a = pd.read_csv(file_a)

        # --- Y 變數設定區 ---
        has_y_2 = st.checkbox("資料集包含目標變數 (Y) 嗎？", value=True, key="hy2")
        if has_y_2:
            col_yb = st.selectbox("Before 的 Y 欄位：", df_b.columns, key="ycb")
            col_ya = st.selectbox("After 的 Y 欄位：", df_a.columns, key="yca")
            # 抽離 Y，以 Before 的 Y 為主保留到最後
            y_baseline = df_b[[col_yb]].copy()
            X_b = df_b.drop(columns=[col_yb])
            X_a = df_a.drop(columns=[col_ya])
        else:
            y_baseline = pd.DataFrame()
            X_b, X_a = df_b.copy(), df_a.copy()

        st.subheader("1. 數據對齊與差異計算 (Diff = After - Before)")
        # 僅對特徵 X 進行對齊與計算
        Before_aligned, After_aligned = X_b.align(X_a, join='inner', axis=None)
        Diff = After_aligned - Before_aligned
        
        st.success(f"✅ 對齊完成！共有 {len(Before_aligned.columns)} 個共同特徵。")

        st.divider()
        st.subheader("2. 差異矩陣平移 (Shift to Non-Negative)")
        min_value = Diff.values.min()
        shift_value_min = int(np.ceil(-min_value)) if min_value < 0 else 0
        Diff_shifted_min = Diff + shift_value_min
        
        shift_value_const = st.number_input("設定常數平移值:", value=100, step=10)
        Diff_shifted_const = Diff + shift_value_const

        # 縫合 Y 變數準備匯出 (透過 index 對齊確保沒對錯)
        if has_y_2:
            y_aligned = y_baseline.loc[Before_aligned.index]
            final_min = pd.concat([y_aligned, Diff_shifted_min], axis=1)
            final_const = pd.concat([y_aligned, Diff_shifted_const], axis=1)
        else:
            final_min, final_const = Diff_shifted_min, Diff_shifted_const

        col_min, col_const = st.columns(2)
        with col_min:
            st.write(f"**A. 最小整數平移 (Shift = {shift_value_min})**")
            st.dataframe(final_min.head(), use_container_width=True)
            st.download_button("📥 下載 Shifted_Min", final_min.to_csv(index=False).encode('utf-8'), "ASV_Diff_Shifted_Min.csv", "text/csv")
            
        with col_const:
            st.write(f"**B. 常數平移 (Shift = {shift_value_const})**")
            st.dataframe(final_const.head(), use_container_width=True)
            st.download_button(f"📥 下載 Shifted_Const{shift_value_const}", final_const.to_csv(index=False).encode('utf-8'), f"ASV_Diff_Shifted_Const{shift_value_const}.csv", "text/csv")
    else:
        st.warning("⚠️ 請同時上傳 Before 與 After 兩個 CSV 檔案。")


# ==========================================
# --- 第三步：NMF 非負矩陣分解 ---
# ==========================================
with tab3:
    st.header("🧠 第三步：NMF 非負矩陣分解")
    
    file_nmf = st.file_uploader("📂 上傳平移後的非負矩陣 CSV", type=["csv"], key="nmf_up")
    
    # --- 初始化 session_state 來記憶最佳 K 值 ---
    if 'best_k' not in st.session_state:
        st.session_state.best_k = 15  # 預設值

    if file_nmf:
        df_nmf_raw = pd.read_csv(file_nmf)
        
        # --- Y 變數設定區 ---
        has_y_3 = st.checkbox("此矩陣包含目標變數 (Y) 嗎？", value=True, key="hy3")
        if has_y_3:
            y_col_3 = st.selectbox("選擇目標變數 (Y) 欄位：", df_nmf_raw.columns, key="yc3")
            df_y_nmf = df_nmf_raw[[y_col_3]].copy()
            df_X_nmf = df_nmf_raw.drop(columns=[y_col_3])
        else:
            df_y_nmf = pd.DataFrame()
            df_X_nmf = df_nmf_raw.copy()

        st.subheader("1. 尋找最佳 K 值 (Elbow Method 自動偵測)")
        max_k = st.slider("測試的最大 K 值", 10, 100, 50, 10)
        
        if st.button("🚀 執行 Elbow Method 運算"):
            with st.spinner(f'計算 K=1 到 {max_k} 中... (這可能需要一到兩分鐘)'):
                errors = []
                K_range = list(range(1, max_k + 1))
                
                for k in K_range:
                    nmf_test = NMF(n_components=k, init='random', max_iter=1000, random_state=0)
                    nmf_test.fit(df_X_nmf)
                    errors.append(nmf_test.reconstruction_err_)
                
                # --- 自動計算最佳 K 值 (基於 Pandas 差分計算) ---
                df_metrics = pd.DataFrame({'K': K_range, 'Reconstruction_Error': errors})
                
                # 1. 一階差分 (誤差下降幅度：代表斜率)
                df_metrics['1st_Derivative (Slope)'] = df_metrics['Reconstruction_Error'].diff()
                
                # 2. 二階差分 (斜率變化率：代表曲率/轉折強烈度)
                df_metrics['2nd_Derivative (Curvature)'] = df_metrics['1st_Derivative (Slope)'].diff()
                
                # 3. 找出二階差分最大值的 Index
                max_curve_idx = df_metrics['2nd_Derivative (Curvature)'].idxmax()
                
                # 【關鍵修正】：因為差分向後相減的特性，數學上的最大變動點會落在轉折的「下一格」
                # 所以我們手動把 Index 減 1，退回真實的手肘轉折點
                best_k_idx = max_curve_idx - 1
                best_k_auto = int(df_metrics.loc[best_k_idx, 'K'])
                
                # 更新 session_state，讓下方的 number_input 自動跟隨
                st.session_state.best_k = best_k_auto
                
                # --- 繪圖並標示最佳 K 值 ---
                fig3, ax3 = plt.subplots(figsize=(10, 5))
                ax3.plot(K_range, errors, marker='o', color='b', label='Reconstruction Error')
                ax3.axvline(x=best_k_auto, color='red', linestyle='--', alpha=0.7, label=f'Auto Best K = {best_k_auto}')
                ax3.scatter(best_k_auto, errors[best_k_idx], color='red', s=100, zorder=5)
                
                ax3.set_title('Elbow Method with Auto-Detection (Max Curvature)')
                ax3.set_xlabel('n_components (K)')
                ax3.set_ylabel('Reconstruction Error')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                st.pyplot(fig3)
                
                st.success(f"✅ 系統分析斜率變動完成！判定誤差曲線在 **K = {best_k_auto}** 處發生最大曲率轉折。")

                with st.expander("📊 查看詳細數值運算表 (點擊展開)"):
                    st.markdown("系統會尋找 `2nd_Derivative` (二階差分/曲率) 最大值的前一格，作為最佳手肘點。")
                    
                    # 客製化上色函數：只把我們選定的 best_k_idx 那一行塗成綠色
                    def highlight_best_row(row):
                        if row.name == best_k_idx:
                            return ['background-color: lightgreen'] * len(row)
                        return [''] * len(row)
                    
                    # 套用客製化上色並顯示
                    st.dataframe(
                        df_metrics.style.apply(highlight_best_row, axis=1), 
                        use_container_width=True
                    )
                    
                    st.download_button(
                        label="📥 下載 Elbow 評估數值表 (CSV)",
                        data=df_metrics.to_csv(index=False).encode('utf-8'),
                        file_name="NMF_Elbow_Evaluation_Metrics.csv",
                        mime="text/csv"
                    )

        st.divider()
        st.subheader("2. 執行最終 NMF 分解")
        
        # 這裡將 value 綁定到 session_state.best_k，實現自動帶入
        chosen_k = st.number_input("決定使用的 K 值 (已自動帶入最佳建議值):", 1, 200, value=st.session_state.best_k)
        
        if st.button(f"⚡ 以 K={chosen_k} 執行分解"):
            with st.spinner('執行 NMF 拆解中...'):
                # (下方保持你原本修改好的 NMF_final 運算與下載邏輯即可，無需更動)
                nmf_final = NMF(n_components=chosen_k, init='nndsvda', max_iter=1000, random_state=42)
                
                W = nmf_final.fit_transform(df_X_nmf)   # shape: (Samples, k)
                H = nmf_final.components_               # shape: (k, Features)
                
                comp_names = [f"C{i}" for i in range(1, chosen_k + 1)]
                
                # 處理 W 矩陣
                NMF_W = pd.DataFrame(W, columns=comp_names)
                if has_y_3:
                    NMF_W = pd.concat([df_y_nmf.reset_index(drop=True), NMF_W], axis=1)
                
                # 處理 H 矩陣
                NMF_H = pd.DataFrame(H, columns=df_X_nmf.columns)
                NMF_H.insert(0, 'Component', comp_names)
                
                st.success("✅ NMF 分解完成！矩陣格式已優化。")
                
                col_w, col_h = st.columns(2)
                with col_w:
                    st.write(f"**W 矩陣 (包含 Y 與 C1~C{chosen_k}):** `shape {NMF_W.shape}`")
                    st.dataframe(NMF_W.head(), use_container_width=True)
                    st.download_button(
                        label="📥 下載 W 矩陣 (CSV)",
                        data=NMF_W.to_csv(index=False).encode('utf-8'),
                        file_name=f"MG_Gene_ASV_Diff_ShiftMin_NMF_K{chosen_k}_W.csv",
                        mime="text/csv"
                    )
                with col_h:
                    st.write(f"**H 矩陣 (特徵基底 C1~C{chosen_k}):** `shape {NMF_H.shape}`")
                    st.dataframe(NMF_H.head(), use_container_width=True)
                    st.download_button(
                        label="📥 下載 H 矩陣 (CSV)",
                        data=NMF_H.to_csv(index=False).encode('utf-8'),
                        file_name=f"MG_Gene_ASV_Diff_ShiftMin_NMF_K{chosen_k}_H.csv",
                        mime="text/csv"
                    )


# ==========================================
# --- 第四步：群組特徵對比與特定成分重構 ---
# ==========================================
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st # 確保 st 有被引入

with tab4:
    st.header("📊 第四步：群組特徵對比、基因解析與特定成分重構")
    st.markdown("一站式完成分析：找出群組間的「共同/獨立」成分 $\\rightarrow$ 萃取並篩選出特定分類的基因 $\\rightarrow$ **直接重構出特定生物信號的目標特徵矩陣！**")
    
    col_w_up, col_h_up = st.columns(2)
    with col_w_up:
        file_w = st.file_uploader("📂 上傳 W 矩陣 (包含 Y 標籤)", type=["csv"], key="w_up")
    with col_h_up:
        file_h = st.file_uploader("📂 上傳 H 矩陣 (特徵基底)", type=["csv"], key="h_up")

    if file_w and file_h:
        df_w = pd.read_csv(file_w)
        df_h = pd.read_csv(file_h)

        if 'Component' in df_h.columns:
            H_matrix = df_h.set_index('Component')
        else:
            H_matrix = df_h.set_index(df_h.columns[0])

        # ==========================================
        # 1. W 矩陣分析 (宏觀分群)
        # ==========================================
        st.divider()
        st.subheader("1. W 矩陣分析：定義群組的「共同」與「獨立」成分")
        
        class_col_target = st.selectbox("請選擇 W 矩陣中的目標變數 (Y) 欄位：", df_w.columns)
        
        if class_col_target:
            top_k_val = st.number_input("設定要觀察每個樣本的「前 K 大」成分 (K=1 代表只看最強成分):", min_value=1, max_value=len(H_matrix), value=1, step=1)
            
            with st.spinner('運算群組對比中...'):
                topk_per_sample, topk_counts, class_comparison = topk_components_class_comparison(df_w, class_col=class_col_target, top_k=top_k_val)
                df_topk_counts = topk_counts.fillna(0).astype(int)
            
            # --- A. 長條圖 (Bar Chart) 取代原有的氣泡圖 ---
            st.markdown(f"**A. 特徵分佈長條圖 (不同群組在各成分的分佈數量)**")
            df_melted = df_topk_counts.reset_index().melt(id_vars=class_col_target, var_name="Component", value_name="Count")
            
            # 為了讓圖表 X 軸排序更美觀，我們抓取 Component 的數字進行排序
            df_melted['Comp_Num'] = df_melted['Component'].apply(lambda x: int(str(x).replace('C', '')) if str(x).startswith('C') and str(x).replace('C', '').isdigit() else 999)
            df_melted = df_melted.sort_values(['Comp_Num', 'Component'])

            fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
            # 使用 sns.barplot 繪製分組長條圖
            sns.barplot(data=df_melted, x="Component", y="Count", hue=class_col_target, palette="Set1", ax=ax_bar)
            ax_bar.set_title(f"Component Counts by {class_col_target} (Top-{top_k_val} per sample)")
            ax_bar.set_xlabel("NMF Component")
            ax_bar.set_ylabel("Count")
            plt.grid(True, axis='y', linestyle='--', alpha=0.5) # 只留橫向格線讓畫面更清爽
            st.pyplot(fig_bar)

            # --- 排序小幫手 ---
            def sort_comps(comps):
                def extract_num(c):
                    try: return int(c.replace('C', '')) if isinstance(c, str) and c.startswith('C') else float('inf')
                    except: return float('inf')
                return sorted(list(comps), key=extract_num)

            # --- B. 標籤區 ---
            st.markdown(f"**B. 群組特徵標籤 (Shared vs Unique)**")
            shared_comps, unique_comps, all_present_comps = set(), set(), set()

            if class_comparison:
                for k, v in class_comparison.items():
                    sorted_v = sort_comps(v)
                    comps_str = ", ".join(sorted_v) if len(v) > 0 else "無"
                    if "Shared" in k:
                        shared_comps.update(v)
                        st.success(f"🤝 **兩組共同成分 (Shared)**: {comps_str}")
                    elif "Unique" in k:
                        unique_comps.update(v)
                        st.warning(f"🏷️ **獨立成分 ({k.replace('Unique_to_Class_', '')})**: {comps_str}")
                    else:
                        all_present_comps.update(v)
                        st.info(f"📊 **{k.replace('Class_', '').replace('_components', '')} (該群體有出現的成分)**: {comps_str}")
            else:
                for cls in df_w[class_col_target].unique():
                    comps = set(topk_counts.loc[cls][topk_counts.loc[cls] > 0].index)
                    all_present_comps.update(comps)

            # ==========================================
            # 2. H 矩陣解析 (選擇成分)
            # ==========================================
            st.divider()
            st.subheader("2. H 矩陣解析：選擇目標成分")
            st.markdown("💡 **操作說明**：請根據上方的分析結果，挑選出您感興趣的特定成分（例如：只選取兩組共用的成分，或某組獨有的成分）。這些被選中的成分將作為下一步基因萃取的基礎庫。")
            
            classes = df_w[class_col_target].unique().tolist()
            radio_options = ["🤝 共同成分 (Shared)", "🏷️ 所有獨立成分 (All Unique)"]
            if len(classes) >= 2:
                c0, c1 = classes[:2]
                radio_options.extend([f"🏷️ 獨立成分 ({c0})", f"🏷️ 獨立成分 ({c1})"])
            radio_options.append("🌐 只要有出現就算 (All Present)")

            comp_filter_option = st.radio("快速帶入條件：", options=radio_options, horizontal=True)
            
            selected_comps_auto = set()
            if "共同" in comp_filter_option: selected_comps_auto = shared_comps
            elif "所有獨立成分" in comp_filter_option: selected_comps_auto = unique_comps
            elif "只要有出現就算" in comp_filter_option: selected_comps_auto = shared_comps | unique_comps | all_present_comps
            elif len(classes) >= 2:
                if f"({c0})" in comp_filter_option:
                    selected_comps_auto = next((val for key, val in class_comparison.items() if "Unique" in key and str(c0) in key), set())
                elif f"({c1})" in comp_filter_option:
                    selected_comps_auto = next((val for key, val in class_comparison.items() if "Unique" in key and str(c1) in key), set())

            final_selected_comps = st.multiselect(
                "✨ 系統已自動帶入成分，請確認或手動調整 (這些成分將進入下一步進行萃取)：",
                options=sort_comps(H_matrix.index.tolist()), 
                default=sort_comps(selected_comps_auto)
            )

            # ==========================================
            # 3. 🚀 目標信號萃取與特徵矩陣重構
            # ==========================================
            if final_selected_comps:
                st.divider()
                st.subheader("3. 🚀 目標信號萃取與特徵矩陣重構 (Signal Extraction & Reconstruction)")
                
                # 將提取前 N 個基因特徵的設定移到這裡
                st.markdown("💡 **操作說明**：設定每個被選中的成分中，要保留多少個最具代表性（權重最高）的基因特徵。接著系統將比對這些特徵並重構出最終矩陣。")
                top_n_val = st.number_input("每個成分要保留前 N 個基因特徵 (輸入 0 匯出全部):", min_value=0, max_value=len(H_matrix.columns), value=15, step=1)
                use_top_n = top_n_val if top_n_val > 0 else None

                with st.spinner('解析 H 矩陣並計算特徵交集中...'):
                    # 依據 N 的設定解析 H 矩陣
                    sorted_features_dict = extract_sorted_features_from_H(H_matrix, top_n=use_top_n)
                    all_features_df = pd.concat(sorted_features_dict, names=['Component', 'DropIndex']).reset_index(level='DropIndex', drop=True).reset_index()
                    
                    # 篩選出使用者選取的成分
                    filtered_features_df = all_features_df[all_features_df['Component'].isin(final_selected_comps)]
                    feat_col_name = 'Feature' if 'Feature' in filtered_features_df.columns else filtered_features_df.columns[1]
                    
                    df_list_for_summary = []
                    for comp in final_selected_comps:
                        df_comp = filtered_features_df[filtered_features_df['Component'] == comp].copy()
                        df_list_for_summary.append(df_comp)
                        
                    summary_df = compare_ranked_features_summary(
                        df_list=df_list_for_summary,
                        df_names=final_selected_comps,
                        feature_col=list(filtered_features_df.columns).index(feat_col_name),
                        percentage_col=list(filtered_features_df.columns).index('Percentage (%)') if 'Percentage (%)' in filtered_features_df.columns else 3
                    )
                    summary_df.index.name = "ASV_Feature"

                    if 'Total_Appearance' not in summary_df.columns:
                        summary_df['Total_Appearance'] = summary_df[final_selected_comps].sum(axis=1)
                    
                    summary_df['Max_Intensity_Score'] = summary_df[final_selected_comps].max(axis=1)

                with st.expander("🔍 檢視 ASV 在各成分間的詳細分佈 (按出現頻次與強度排序)", expanded=True):
                    display_summary = summary_df.copy()
                    display_summary = display_summary.sort_values(
                        by=['Total_Appearance', 'Max_Intensity_Score'], 
                        ascending=[False, False]
                    )
                    
                    show_cols = final_selected_comps + ['Total_Appearance']
                    plot_ready_df = display_summary[show_cols]

                    def style_summary_table(df):
                        s = df.style
                        for col in final_selected_comps:
                            s = s.map(
                                lambda v: 'background-color: #1f77b4; color: white; font-weight: bold;' if v > 0 else 'color: #d3d3d3;',
                                subset=[col]
                            )
                        s = s.background_gradient(cmap='Blues', subset=['Total_Appearance'])
                        return s

                    st.dataframe(style_summary_table(plot_ready_df), use_container_width=True, height=400)
                    st.caption(f"📊 排序說明：優先顯示多組共有特徵 (Shared)，並結合該特徵在各 Component 的出現權重排列。")

                    # 下載 display_summary
                    csv_summary = display_summary.to_csv(index=True).encode('utf-8-sig')
                    st.download_button(
                        label="📥 下載詳細分佈表 (CSV)",
                        data=csv_summary,
                        file_name="ASV_Feature_Summary.csv",
                        mime="text/csv",
                        key="download_summary_btn" 
                    )

                # --- 策略卡片與重構執行 ---
                st.markdown("🎯 請選擇重構策略：")
                c1, c2, c3 = st.columns(3)
                with c1:
                    count_all = len(summary_df)
                    st.info(f"🌐 **保留所有特徵**\n\n共 {count_all} 個 ASVs")
                with c2:
                    count_unique = len(summary_df[summary_df["Total_Appearance"] == 1])
                    st.warning(f"🏷️ **專屬生物特徵 (Exclusive)**\n\n共 {count_unique} 個 ASVs")
                with c3:
                    count_shared = len(summary_df[summary_df.get("Shared_in_All", summary_df["Total_Appearance"] == len(final_selected_comps)) == 1])
                    st.success(f"🤝 **核心共用特徵 (Core Shared)**\n\n共 {count_shared} 個 ASVs")

                strategy = st.radio("選擇萃取策略：", options=["🌐 保留所有特徵 (預設)", "🏷️ 僅保留專屬生物特徵", "🤝 僅保留核心共用特徵"], horizontal=True)

                if st.button("⚡ 執行目標矩陣重構 (Reconstruct)"):
                    with st.spinner("執行局部內積重構運算中..."):
                        # 💡 修正 1：確保判斷字眼與 Radio 選項一致
                        if "專屬生物" in strategy:
                            final_asv_summary = summary_df[summary_df["Total_Appearance"] == 1]
                        elif "核心共用" in strategy:
                            final_asv_summary = summary_df[summary_df.get("Shared_in_All", summary_df["Total_Appearance"] == len(final_selected_comps)) == 1]
                        else:
                            final_asv_summary = summary_df
                            
                        if len(final_asv_summary) == 0:
                            st.error(f"⚠️ 在此策略下找不到符合的 ASV。")
                        else:
                            target_asvs = final_asv_summary.index.tolist()
                            def natural_sort_key(s):
                                return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]
                            
                            valid_asvs = sorted([asv for asv in target_asvs if asv in H_matrix.columns], key=natural_sort_key)
                            valid_w_comps = [c for c in final_selected_comps if c in df_w.columns]
                            
                            W_sub = df_w[valid_w_comps].values
                            H_sub = H_matrix.loc[valid_w_comps, valid_asvs].values
                            V_reconstructed = np.dot(W_sub, H_sub)
                            
                            df_final = pd.DataFrame(V_reconstructed, columns=valid_asvs)
                            df_final.insert(0, class_col_target, df_w[class_col_target].values)
                            df_final.index = df_w.index
                            
                            st.session_state.reconstructed_df = df_final
                            st.session_state.current_strategy = strategy
                            st.success(f"✅ 重構成功！已產生 {len(valid_asvs)} 個特徵的矩陣。")

                # --- 顯示重構後的數據預覽與下載 ---
                if 'reconstructed_df' in st.session_state:
                    recon_df = st.session_state.reconstructed_df
                    strat = st.session_state.get('current_strategy', '')
                    
                    st.divider()
                    st.subheader(f"📋 原始數據預覽與下載 ({strat})")
                    
                    st.dataframe(recon_df.head(15).style.background_gradient(subset=recon_df.columns[1:], cmap='BuPu'), use_container_width=True)
                    
                    # 1. 解析策略名稱，轉換為適合檔名的文字
                    # 💡 修正 2：確保檔名生成邏輯也能抓到正確的字眼
                    if "所有特徵" in strat:
                        strat_name = "All"
                    elif "專屬生物" in strat:
                        strat_name = "Exclusive"
                    elif "核心共用" in strat:
                        strat_name = "CoreShared"
                    else:
                        strat_name = "Custom"
                    
                    # 2. 處理成分名稱 (如果選太多個，就只顯示數量避免檔名過長)
                    if len(final_selected_comps) <= 5:
                        comps_str = "_".join(sort_comps(final_selected_comps))
                    else:
                        comps_str = f"{len(final_selected_comps)}Comps"
                        
                    # 3. 取得特徵數量 (總欄位數減去 1 個 Y 標籤欄位)
                    feature_count = recon_df.shape[1] - 1
                    
                    # 4. 組合動態檔名
                    dynamic_filename = f"Reconstructed_{comps_str}_{strat_name}_{feature_count}ASVs.csv"
                    
                    csv_data = recon_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label=f"📥 下載重構矩陣 ({dynamic_filename})", 
                        data=csv_data, 
                        file_name=dynamic_filename, 
                        mime="text/csv"
                    )
            else:
                st.warning("⚠️ 請從步驟 2 選擇至少一個成分以繼續。")

with tab5:
    st.header("🤖 第五步：機器學習模型訓練與驗證 (LOOCV)")
    st.markdown("上傳您的特徵重構矩陣，系統將自動進行特徵縮放，並使用 Leave-One-Out 與 GridSearch 訓練您選擇的模型。")

    # --- 1. 檔案上傳與資料準備 ---
    file_model_data = st.file_uploader("📂 上傳重構後的特徵資料 (CSV，需包含 Y 標籤)", type=["csv"], key="model_data_up")

    if file_model_data:
        df = pd.read_csv(file_model_data)
        
        # 讓使用者選擇 Y 標籤欄位
        default_y = 'Y' if 'Y' in df.columns else df.columns[0]
        y_col = st.selectbox("請確認您的目標變數 (Y) 欄位：", df.columns, index=list(df.columns).index(default_y))
        
        st.write("📊 預覽上傳資料：")
        st.dataframe(df.head(), use_container_width=True)

        st.divider()
        st.subheader("⚙️ 模型設定")
        
        available_models = ["XGBoost", "Random Forest", "Lasso (L1 Logistic)"]
        selected_models = st.multiselect(
            "請選擇要訓練的模型 (可多選)：", 
            options=available_models, 
            default=available_models
        )

        # ==========================================
        # --- 模型訓練區塊 (按下按鈕才執行，執行完存入大腦) ---
        # ==========================================
        if st.button("🚀 開始訓練模型 (這可能需要幾分鐘時間)"):
            if not selected_models:
                st.warning("⚠️ 請至少選擇一個模型來進行訓練！")
            else:
                with st.spinner("資料前處理與模型訓練中，請耐心等候..."):
                    
                    # 資料前處理
                    X_raw = df.drop(y_col, axis=1).copy()
                    Y_raw = df[y_col].copy().values
                    
                    MMscaler = MinMaxScaler(feature_range=(0, 1))
                    X_normalized = MMscaler.fit_transform(X_raw)
                    X_df = pd.DataFrame(data=X_normalized, columns=X_raw.columns)
                    
                    # 模型與參數設定
                    all_models_setup = {
                        "XGBoost": (xgb.XGBClassifier(missing=np.nan, random_state=4), 
                                    {'max_depth': [4, 5, 6], 'gamma': [0, 0.25, 1.0], 'scale_pos_weight': [1, 3, 5]}),
                        "Random Forest": (RandomForestClassifier(random_state=4), 
                                          {'max_depth': [None, 5, 10], 'n_estimators': [50, 100]}),
                        "Lasso (L1 Logistic)": (LogisticRegression(penalty='l1', solver='liblinear', random_state=4), 
                                                {'C': [0.1, 1.0, 10.0]})
                    }
                    
                    models_setup = {k: v for k, v in all_models_setup.items() if k in selected_models}
                    cross_val = LeaveOneOut()
                    
                    # 💡 確保這個變數有被宣告
                    all_model_results = {}
                    
                    progress_bar = st.progress(0)
                    total_models = len(models_setup)
                    
                    # 開始迴圈訓練
                    for m_idx, (model_name, (clf)) in enumerate(models_setup.items()):
                        st.toast(f"正在訓練 {model_name}...")
                        
                        each_round_y_probability = []
                        each_round_y_prediction = []
                        
                        for i in range(len(X_df)):
                            X_train = X_df.drop(i, axis=0)
                            Y_train = np.delete(Y_raw, i)
                            X_test = X_df.iloc[[i]]
                            
                            # 💡 這裡的 n_jobs 已經幫你設定為 1，避免 Windows 報錯
                            gridS_model = GridSearchCV(
                                estimator=clf,
                                param_grid=param_grid,
                                scoring='accuracy',
                                n_jobs=1,
                                cv=cross_val, 
                                verbose=0,
                                refit=True
                            )
                            
                            gridS_model.fit(X_train, Y_train) 
                            train_model = gridS_model.best_estimator_
                            
                            y_prob_pred = train_model.predict_proba(X_test)[0, 1] 
                            prediction = train_model.predict(X_test)[0]
                            
                            each_round_y_probability.append(y_prob_pred)
                            each_round_y_prediction.append(prediction)
                            
                            current_step = m_idx * len(X_df) + (i + 1)
                            total_steps = len(X_df) * total_models
                            progress_bar.progress(current_step / total_steps)

                        # 計算指標
                        fpr, tpr, thresholds = roc_curve(Y_raw, each_round_y_probability, pos_label=1)
                        J_score = tpr - fpr
                        best_JTH_cutpoint_index = np.argmax(J_score)
                        best_JS_threshold = thresholds[best_JTH_cutpoint_index]
                        
                        final_prediction = (np.array(each_round_y_probability) >= best_JS_threshold).astype('int')
                        confusion = confusion_matrix(Y_raw, final_prediction)
                        
                        TP = confusion[1, 1] if confusion.shape == (2, 2) else 0
                        TN = confusion[0, 0] if confusion.shape == (2, 2) else 0
                        FP = confusion[0, 1] if confusion.shape == (2, 2) else 0
                        FN = confusion[1, 0] if confusion.shape == (2, 2) else 0
                        
                        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
                        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
                        specificity = TN / (FP + TN) if (FP + TN) > 0 else 0
                        f1score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
                        
                        try:
                            auc_value = roc_auc_score(Y_raw, each_round_y_probability)
                        except ValueError:
                            auc_value = 0 
                        
                        result_df = pd.DataFrame({
                            'y_true': Y_raw,
                            'y_pred_original': each_round_y_prediction,
                            'y_prob': each_round_y_probability,
                            'y_pred_adjusted': final_prediction
                        })
                        
                        # 把每個模型的結果存進字典
                        all_model_results[model_name] = {
                            "metrics": {
                                "Accuracy (%)": round(accuracy * 100, 2),
                                "AUC (%)": round(auc_value * 100, 2),
                                "F1 Score (%)": round(f1score * 100, 2),
                                "Sensitivity (%)": round(sensitivity * 100, 2),
                                "Specificity (%)": round(specificity * 100, 2),
                                "Precision (%)": round(precision * 100, 2),
                                "Threshold": round(best_JS_threshold, 4)
                            },
                            "confusion": confusion,
                            "predictions": result_df
                        }

                    progress_bar.empty()
                    st.success("✅ 所有選定模型訓練與評估完成！請在下方查看結果。")
                    
                    # 💡 訓練完畢！把結果寫入 session_state (暫存記憶)
                    st.session_state['trained_results'] = all_model_results
                    st.session_state['trained_models_keys'] = list(models_setup.keys())

        # ==========================================
        # --- 結果顯示區塊 (直接從大腦讀取，避免重跑) ---
        # ==========================================
        if 'trained_results' in st.session_state:
            # 從暫存記憶中把資料拿出來
            saved_results = st.session_state['trained_results']
            saved_keys = st.session_state['trained_models_keys']
            
            st.divider()
            
            # 建立比較用 DataFrame
            comparison_data = []
            for name, res in saved_results.items():
                row = res["metrics"].copy()
                row["Model"] = name
                comparison_data.append(row)
            df_compare = pd.DataFrame(comparison_data).set_index("Model")

            # 設定 Tabs
            tab_names = ["🏆 綜合比較"] + saved_keys
            tabs = st.tabs(tab_names)
            
            # --- Tab 0: 綜合比較 ---
            with tabs[0]:
                st.subheader("📊 模型效能總覽")
                
                metric_cols = [col for col in df_compare.columns if col != 'Threshold']
                styled_df = df_compare.style.highlight_max(subset=metric_cols, color='lightgreen')
                st.dataframe(styled_df, use_container_width=True)
                
                csv_compare = df_compare.to_csv().encode('utf-8-sig')
                st.download_button(
                    label="📥 下載指標比較表 (CSV)",
                    data=csv_compare,
                    file_name="Models_Comparison.csv",
                    mime="text/csv",
                    key="download_compare_btn"
                )
                
                st.markdown("**指標視覺化比較**")
                df_melted = df_compare.reset_index().melt(id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Score")
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="viridis", ax=ax)
                ax.set_title("Performance Metrics Comparison across Models")
                ax.set_ylabel("Score (%)")
                ax.set_ylim(0, 105) 
                plt.xticks(rotation=45)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                st.pyplot(fig)

            # --- 後續 Tabs: 各別模型詳細資料 ---
            for idx, model_name in enumerate(saved_keys):
                res = saved_results[model_name]
                
                with tabs[idx+1]:
                    st.subheader(f"📈 {model_name} 結果報表")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{res['metrics']['Accuracy (%)']}%")
                    m2.metric("AUC", f"{res['metrics']['AUC (%)']}%")
                    m3.metric("F1 Score", f"{res['metrics']['F1 Score (%)']}%")
                    m4.metric("最佳 Threshold", f"{res['metrics']['Threshold']}")
                    
                    m5, m6, m7, m8 = st.columns(4)
                    m5.metric("Sensitivity", f"{res['metrics']['Sensitivity (%)']}%")
                    m6.metric("Specificity", f"{res['metrics']['Specificity (%)']}%")
                    m7.metric("Precision", f"{res['metrics']['Precision (%)']}%")
                    
                    st.divider()
                    col_conf, col_pred = st.columns([1, 2])
                    
                    with col_conf:
                        st.markdown("**混淆矩陣**")
                        st.dataframe(pd.DataFrame(res['confusion'], columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]), use_container_width=True)
                        
                    with col_pred:
                        st.markdown("**預測明細**")
                        st.dataframe(res['predictions'], use_container_width=True, height=200)
                        
                    # 💡 這裡包裝了加強版的 CSV (附帶混淆矩陣)
                    export_df = res['predictions'].copy()
                    export_df[' | '] = '' 
                    export_df['Confusion Matrix'] = ''
                    export_df['Pred 0'] = np.nan
                    export_df['Pred 1'] = np.nan
                    
                    cm = res['confusion']
                    if len(export_df) >= 2 and cm.shape == (2, 2):
                        export_df.loc[0, 'Confusion Matrix'] = 'True 0'
                        export_df.loc[0, 'Pred 0'] = int(cm[0, 0])
                        export_df.loc[0, 'Pred 1'] = int(cm[0, 1])
                        
                        export_df.loc[1, 'Confusion Matrix'] = 'True 1'
                        export_df.loc[1, 'Pred 0'] = int(cm[1, 0])
                        export_df.loc[1, 'Pred 1'] = int(cm[1, 1])
                    
                    csv = export_df.to_csv(index=False).encode('utf-8-sig')
                    
                    st.download_button(
                        label=f"📥 下載 {model_name} 預測明細與混淆矩陣",
                        data=csv,
                        file_name=f"{model_name}_LOOCV_Results.csv",
                        mime="text/csv",
                        key=f"download_{model_name}_btn"
                    )