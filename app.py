# ============================================================
# 食物营养成分分析 · Streamlit 网站
# ============================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager as fm
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, silhouette_score)
import warnings
warnings.filterwarnings('ignore')

# ---------- 1. 基础路径（本地/云端都适用） ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, 'NotoSansSC-Regular.otf')
DATA_PATH = os.path.join(BASE_DIR, 'cleaned_nutrition_dataset_per100g.csv')

# ---------- 2. 中文字体注册 ----------
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    cn_font = fm.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams['font.sans-serif'] = [cn_font]
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid", {
        "font.sans-serif": [cn_font],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
    })

# ---------- 3. 页面配置 ----------
st.set_page_config(page_title="食物营养分析", page_icon="🥗", layout="wide")
st.title("🥗 食物营养成分分析平台")
st.caption("基于 NHI 营养健康指数的探索式分析 · 聚类画像 · 健康食品推荐")
with st.expander("📖 什么是 NHI 营养健康指数？（点开查看评分规则）", expanded=False):
    st.markdown("""
### NHI = Nutrition Health Index（营养健康指数）

**本项目的核心创新**：把 11 个营养成分压缩成一个 **0~100 分**的健康分数，
让普通读者不用看懂复杂的营养表，也能快速判断一种食物是否健康。

---

#### 📐 打分规则

NHI 从 **50 分**起步，然后：

**✅ 加分项（营养丰富度，最多 +65 分）**

| 成分 | 加分公式 | 上限 |
|---|---|---|
| 蛋白质 | 每克 +1.5 分 | +20 |
| 膳食纤维 | 每克 +1.0 分 | +15 |
| 维生素C | 每毫克 +0.05 分 | +10 |
| 钙 | 每毫克 +0.02 分 | +10 |
| 铁 | 每毫克 +0.5 分 | +10 |

**❌ 扣分项（过量危害，最多 -40 分）**

| 成分 | 扣分公式 | 上限 |
|---|---|---|
| 糖 | 每克 -0.3 分 | -15 |
| 钠（盐） | 每毫克 -0.005 分 | -10 |
| 脂肪 | 每克 -0.2 分 | -15 |

---

#### 📊 分数解读

| 分数区间 | 健康等级 | 典型食物 |
|---|---|---|
| **80 ~ 100** | ⭐⭐⭐⭐⭐ 非常健康 | 深色蔬菜、豆类、白肉 |
| **65 ~ 80** | ⭐⭐⭐⭐ 较为健康 | 水果、全谷物、乳制品 |
| **50 ~ 65** | ⭐⭐⭐ 一般 | 白米饭、普通肉类 |
| **35 ~ 50** | ⭐⭐ 需注意 | 油炸食品、甜食 |
| **0 ~ 35** | ⭐ 建议少吃 | 纯油脂、高糖饮品 |

---

#### 💡 为什么不用单纯看热量？

因为热量只告诉你"有多少能量"，不告诉你"营养质量如何"。

例如：
- 100g 坚果：约 600 kcal，但富含蛋白质 + 健康脂肪 → **NHI ≈ 65**
- 100g 白砂糖：约 400 kcal，几乎全是糖 → **NHI ≈ 40**

**热量低 ≠ 健康，热量高 ≠ 不健康** —— NHI 提供了更立体的评价视角。

---

#### ⚠️ 指标局限

NHI 是基于营养成分的**简化模型**，未考虑：
- 烹饪方式（蒸 vs 炸）
- 食物多样性（单吃 vs 搭配）
- 个体差异（年龄、运动、疾病）

**仅作营养评估的参考工具，不作为医学建议。**
    """)

# ============================================================
# 数据加载与预处理（用 @st.cache_data 缓存，避免每次刷新重算）
# ============================================================
@st.cache_data
def load_and_prepare(path):
    rename_dict = {
        'Vitamin C (mg per 100g)': '维生素C(mg)',
        'Vitamin B11 (mg per 100g)': '维生素B11(mg)',
        'Sodium (mg per 100g)': '钠(mg)',
        'Calcium (mg per 100g)': '钙(mg)',
        'Carbohydrates (g per 100g)': '碳水化合物(g)',
        'food': '食物名称',
        'Iron (mg per 100g)': '铁(mg)',
        'Calories (kcal per 100g)': '热量(kcal)',
        'Sugars (g per 100g)': '糖(g)',
        'Dietary Fiber (g per 100g)': '膳食纤维(g)',
        'Fat (g per 100g)': '脂肪(g)',
        'Protein (g per 100g)': '蛋白质(g)',
        'food_normalized': '食物标准名称'
    }
    df = pd.read_csv(path).rename(columns=rename_dict)

    # 清洗
    df = df.dropna(subset=['食物标准名称'])
    df = df[df['热量(kcal)'] >= 0]
    num_cols = ['维生素C(mg)', '维生素B11(mg)', '钠(mg)', '钙(mg)',
                '碳水化合物(g)', '铁(mg)', '热量(kcal)', '糖(g)',
                '膳食纤维(g)', '脂肪(g)', '蛋白质(g)']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=num_cols).reset_index(drop=True)
    df = df.drop_duplicates(subset=['食物标准名称']).reset_index(drop=True)

    # 食物类别
    food_categories = {
        '油脂类': ['oil', 'margarine', 'lard', 'shortening', 'tallow', 'ghee'],
        '乳制品': ['cheese', 'yogurt', 'yoghurt', 'milk', 'cream', 'custard', 'buttermilk'],
        '肉禽类': ['beef', 'pork', 'chicken', 'lamb', 'mutton', 'meat', 'bacon', 'ham',
                  'sausage', 'turkey', 'duck', 'veal', 'emu', 'ostrich', 'bison',
                  'caribou', 'poultry', 'drumstick', 'steak', 'brisket', 'sirloin'],
        '水产类': ['fish', 'salmon', 'tuna', 'shrimp', 'crab', 'lobster', 'clam',
                  'oyster', 'mussel', 'squid', 'octopus', 'cod', 'trout', 'sardine',
                  'anchovy', 'herring', 'mackerel', 'halibut', 'bass', 'perch',
                  'scallop', 'prawn', 'crayfish', 'eel', 'catfish', 'tilapia',
                  'snapper', 'grouper', 'flounder', 'haddock', 'pollock', 'roe'],
        '谷物制品': ['bread', 'rice', 'pasta', 'noodle', 'spaghetti', 'macaroni',
                   'flour', 'cereal', 'oat', 'bagel', 'cracker', 'toast', 'bun',
                   'roll', 'pita', 'tortilla', 'muffin', 'waffle', 'pancake',
                   'biscuit', 'croissant', 'pretzel', 'couscous', 'quinoa',
                   'barley', 'bulgur', 'millet', 'popcorn'],
        '蔬菜类': ['tomato', 'potato', 'carrot', 'broccoli', 'spinach', 'cabbage',
                  'onion', 'pepper', 'lettuce', 'cucumber', 'squash', 'kale',
                  'cauliflower', 'celery', 'asparagus', 'artichoke', 'beet',
                  'radish', 'turnip', 'leek', 'mushroom', 'pumpkin', 'zucchini',
                  'eggplant', 'okra', 'gourd', 'yam', 'taro', 'cassava', 'lotus',
                  'bamboo', 'fennel', 'shallot', 'chive', 'garlic', 'pigeon'],
        '水果类': ['apple', 'banana', 'orange', 'grape', 'berry', 'mango', 'peach',
                  'pear', 'melon', 'cherry', 'lemon', 'lime', 'fruit', 'strawberry',
                  'blueberry', 'raspberry', 'blackberry', 'cranberry', 'pineapple',
                  'papaya', 'kiwi', 'plum', 'apricot', 'fig', 'date', 'coconut',
                  'avocado', 'pomegranate', 'persimmon', 'guava', 'lychee',
                  'tangerine', 'grapefruit', 'nectarine', 'currant', 'gooseberry',
                  'quince', 'tamarind', 'durian', 'sapote'],
        '蛋类': ['egg'],
        '坚果豆类': ['nut', 'seed', 'almond', 'walnut', 'peanut', 'cashew', 'pecan',
                    'pistachio', 'hazelnut', 'chestnut', 'macadamia', 'sesame',
                    'sunflower', 'flax', 'chia', 'poppy', 'tofu', 'tempeh', 'soy',
                    'lentil', 'chickpea', 'bean', 'pea', 'hummus', 'falafel'],
        '甜食零食': ['sugar', 'candy', 'chocolate', 'syrup', 'cake', 'pie', 'pudding',
                   'ice cream', 'cookie', 'donut', 'pastry', 'honey', 'jam', 'jelly',
                   'frosting', 'fudge', 'caramel', 'marshmallow', 'ladoo', 'burfi',
                   'halwa', 'kheer', 'gulab', 'rasgulla', 'chikki'],
        '饮料酒水': ['soda', 'juice', 'coffee', 'tea', 'drink', 'beer', 'wine',
                   'water', 'cola', 'lemonade', 'smoothie', 'shake', 'nectar',
                   'squash', 'whiskey', 'vodka', 'rum', 'gin', 'lassi'],
        '调味酱料': ['sauce', 'dressing', 'mayonnaise', 'ketchup', 'mustard',
                   'vinegar', 'gravy', 'salsa', 'dip', 'chutney', 'pickle',
                   'seasoning', 'spice', 'herb', 'salt', 'masala', 'powder'],
    }

    def classify_food(name):
        if pd.isna(name):
            return '其他'
        n = str(name).lower()
        for cat, kws in food_categories.items():
            for kw in kws:
                if kw in n:
                    return cat
        return '其他'

    df['食物类别'] = df['食物标准名称'].apply(classify_food)

    # NHI
    def calc_nhi(row):
        score = 50.0
        score += min(row['蛋白质(g)'] * 1.5, 20)
        score += min(row['膳食纤维(g)'] * 1.0, 15)
        score += min(row['维生素C(mg)'] * 0.05, 10)
        score += min(row['钙(mg)'] * 0.02, 10)
        score += min(row['铁(mg)'] * 0.5, 10)
        score -= min(row['糖(g)'] * 0.3, 15)
        score -= min(row['钠(mg)'] * 0.005, 10)
        score -= min(row['脂肪(g)'] * 0.2, 15)
        return max(0.0, min(100.0, score))

    df['营养健康指数NHI'] = df.apply(calc_nhi, axis=1)

    total_macro = df['碳水化合物(g)'] + df['蛋白质(g)'] + df['脂肪(g)']
    df['碳水占比'] = df['碳水化合物(g)'] / (total_macro + 1e-6)
    df['蛋白占比'] = df['蛋白质(g)'] / (total_macro + 1e-6)
    df['脂肪占比'] = df['脂肪(g)'] / (total_macro + 1e-6)

    return df


# 加载数据
try:
    df = load_and_prepare(DATA_PATH)
except FileNotFoundError:
    st.error(f"❌ 找不到数据文件：{DATA_PATH}")
    st.stop()

# 全局共用特征列表
cluster_features = ['热量(kcal)', '碳水化合物(g)', '蛋白质(g)', '脂肪(g)',
                    '糖(g)', '膳食纤维(g)', '钠(mg)', '钙(mg)',
                    '铁(mg)', '维生素C(mg)']

# ============================================================
# 页面导航
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 数据概览", "📈 营养分布", "🧬 聚类画像",
     "🤖 模型解读", "🏆 健康排行", "🔍 食物查询"]
)


# ---------- Tab 1: 数据概览 ----------
with tab1:
    st.header("📊 数据概览")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("食物总数", f"{len(df):,}")
    c2.metric("食物类别", df['食物类别'].nunique())
    c3.metric("平均 NHI", f"{df['营养健康指数NHI'].mean():.1f}")
    c4.metric("平均热量", f"{df['热量(kcal)'].mean():.0f} kcal")

    st.subheader("数据样例")
    st.dataframe(df[['食物标准名称', '食物类别', '热量(kcal)',
                     '蛋白质(g)', '脂肪(g)', '营养健康指数NHI']].head(20))

    st.subheader("类别分布")
    st.bar_chart(df['食物类别'].value_counts())

    st.subheader("数值字段描述统计")
    st.dataframe(df[cluster_features + ['营养健康指数NHI']].describe().T.round(2))
    st.divider()
    st.subheader("🏅 NHI 分数段分布")

    # 分数段划分
    bins = [0, 35, 50, 65, 80, 100]
    labels = [' 建议少吃\n(0~35)',
              ' 需注意\n(35~50)',
              ' 一般\n(50~65)',
              ' 较为健康\n(65~80)',
              ' 非常健康\n(80~100)']
    df['NHI等级'] = pd.cut(df['营养健康指数NHI'], bins=bins,
                            labels=labels, include_lowest=True)
    level_count = df['NHI等级'].value_counts().reindex(labels, fill_value=0)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        colors = ['#D62728', '#FF7F0E', '#FFD700', '#77DD77', '#2CA02C']
        bars = ax.barh(range(len(level_count)), level_count.values,
                       color=colors, edgecolor='gray')
        ax.set_yticks(range(len(level_count)))
        ax.set_yticklabels(level_count.index, fontsize=10)
        for i, v in enumerate(level_count.values):
            ax.text(v + 5, i, f'{v} 种', va='center', fontsize=10)
        ax.set_xlabel('食物数量')
        ax.set_title('NHI 分数段分布', fontsize=12)
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.markdown("""
        #### 📌 如何阅读

        **当前数据集概况**：

        - 全库 **3432** 种食物平均 NHI = **58.0**
        - 大部分食物落在 **50~65** 区间（⭐⭐⭐ 一般）
        - 分数段越高，代表营养质量越好、加工越少

        #### 🎯 应用建议

        | 使用场景 | 推荐分数 |
        |---|---|
        | 日常主食 | ≥ 55 |
        | 减脂餐 | ≥ 65 |
        | 增肌/健身 | ≥ 70 |
        | 老人/病人膳食 | ≥ 75 |

        > 💡 分数不是越高越好，而是"越高越适合作为主要食物"。
        > 偶尔吃 NHI 低的食物（如蛋糕）完全可以，
        > 关键是**整体饮食结构**是否健康。
        """)


# ---------- Tab 2: 营养分布 ----------
with tab2:
    st.header("📈 营养分布与相关性")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['热量(kcal)'], bins=60, color='skyblue', kde=True, ax=ax)
        ax.set_title('食物热量分布')
        ax.set_xlabel('热量（kcal/100g）')
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        order = df.groupby('食物类别')['热量(kcal)'].median().sort_values().index
        sns.boxplot(x='食物类别', y='热量(kcal)', data=df, order=order,
                    palette='Set2', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title('各类食物热量分布')
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("营养特征相关性热力图")
    corr_cols = ['热量(kcal)', '碳水化合物(g)', '蛋白质(g)', '脂肪(g)',
                 '糖(g)', '膳食纤维(g)', '钠(mg)', '钙(mg)',
                 '铁(mg)', '维生素C(mg)', '营养健康指数NHI']
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title('营养特征相关性热力图')
    st.pyplot(fig)
    plt.close(fig)
    st.info("**结论**：热量与脂肪强正相关（r≈0.9），NHI 与蛋白质/纤维正相关，与脂肪/糖负相关。")


# ---------- Tab 3: 聚类画像 ----------
with tab3:
    st.header("🧬 KMeans 聚类画像（K=5）")

    @st.cache_resource
    def run_cluster(data):
        X = np.log1p(data[cluster_features].copy())
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        km = KMeans(n_clusters=5, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        pca = PCA(n_components=2, random_state=42)
        Xp = pca.fit_transform(Xs)
        return labels, Xp, pca.explained_variance_ratio_.sum()

    labels, Xp, var_ratio = run_cluster(df)
    df_v = df.copy()
    df_v['聚类编号'] = labels
    df_v['PC1'], df_v['PC2'] = Xp[:, 0], Xp[:, 1]

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='聚类编号', data=df_v,
                    palette='Set2', s=25, alpha=0.7, ax=ax)
    ax.set_title(f'PCA 降维聚类（累计方差解释率 {var_ratio:.1%}）')
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("各聚类营养特征均值")
    profile = df_v.groupby('聚类编号')[cluster_features + ['营养健康指数NHI']].mean().round(2)
    st.dataframe(profile)

    st.info("**结论**：5 个簇分别对应蛋白主导型、高糖盐型、营养贫瘠型、碳水主导型、综合均衡型。")


# ---------- Tab 4: 模型解读 ----------
with tab4:
    st.header("🤖 营养特征 vs 食物类别：关联强度分析")
    st.caption("本页不是预测任务，而是验证营养特征与类别的关联结构。")

    @st.cache_resource
    def run_models(data):
        valid_cats = data['食物类别'].value_counts()
        valid_cats = valid_cats[valid_cats >= 20].index
        sub = data[data['食物类别'].isin(valid_cats)].copy()
        le = LabelEncoder()
        sub['label'] = le.fit_transform(sub['食物类别'])

        feats = cluster_features + ['营养健康指数NHI', '碳水占比', '蛋白占比', '脂肪占比']
        X, y = sub[feats].values, sub['label'].values
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y)

        sc = StandardScaler()
        Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)

        lr = LogisticRegression(max_iter=3000, class_weight='balanced',
                                C=0.5, random_state=42)
        rf = RandomForestClassifier(n_estimators=300, max_depth=20,
                                    class_weight='balanced',
                                    random_state=42, n_jobs=1)
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                            max_depth=8, random_state=42)

        lr.fit(Xtr_s, ytr); pred_lr = lr.predict(Xte_s)
        rf.fit(Xtr, ytr);   pred_rf = rf.predict(Xte)
        gb.fit(Xtr, ytr);   pred_gb = gb.predict(Xte)

        results = pd.DataFrame({
            '模型': ['逻辑回归', '随机森林', '梯度提升'],
            '准确率': [accuracy_score(yte, pred_lr),
                       accuracy_score(yte, pred_rf),
                       accuracy_score(yte, pred_gb)]
        }).sort_values('准确率', ascending=False)

        preds = {'逻辑回归': pred_lr, '随机森林': pred_rf, '梯度提升': pred_gb}
        best = results.iloc[0]['模型']
        return results, le, yte, preds[best], best, rf, feats

    with st.spinner("正在训练模型……（首次约 1 分钟）"):
        results, le, y_test, best_pred, best_name, rf, feats = run_models(df)

    st.subheader("模型准确率对比")
    st.dataframe(results.style.format({'准确率': '{:.4f}'}))

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(results['模型'], results['准确率'],
                  color=['#4C72B0', '#55A868', '#C44E52'])
    for b, s in zip(bars, results['准确率']):
        ax.text(b.get_x() + b.get_width()/2, s + 0.005, f'{s:.4f}', ha='center')
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
    ax.set_ylim(0, 1.05); ax.set_ylabel('准确率')
    ax.set_title('模型准确率对比（随机基线 0.5）')
    st.pyplot(fig); plt.close(fig)

    st.subheader(f"最佳模型（{best_name}）混淆矩阵")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(confusion_matrix(y_test, best_pred), annot=True, fmt='d',
                cmap='Blues', xticklabels=le.classes_,
                yticklabels=le.classes_, ax=ax)
    ax.set_xlabel('预测类别'); ax.set_ylabel('真实类别')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig); plt.close(fig)

    st.subheader("特征重要性（随机森林）")
    imp = pd.Series(rf.feature_importances_, index=feats).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind='barh', color='teal', ax=ax)
    ax.set_xlabel('重要性得分')
    st.pyplot(fig); plt.close(fig)

    st.info("""
    **结论**：
    - 非线性模型（RF / GBDT）明显优于线性模型（LR），说明关联存在非线性成分
    - 但最高准确率仅约 0.5，说明**食物类别不能仅由营养决定**
    - 主导特征：热量、脂肪、蛋白质
    - 本页价值：揭示关联结构，而非追求预测精度
    """)


# ---------- Tab 5: 健康排行 ----------
with tab5:
    st.header("🏆 健康食品排行")
    st.caption("以下是 NHI > 70 且热量 < 200 kcal 的食物，"
               "可作为日常饮食的优先选择（NHI 规则见首页 📖 面板）")
    st.subheader("Top 20 低卡高 NHI 食品")
    top_healthy = df[(df['热量(kcal)'] < 200) &
                     (df['营养健康指数NHI'] > 70)] \
                    .sort_values('营养健康指数NHI', ascending=False).head(20)
    if len(top_healthy) == 0:
        st.warning("当前筛选条件下没有符合条件的食物")
    else:
        st.dataframe(top_healthy[['食物标准名称', '食物类别', '热量(kcal)',
                                  '蛋白质(g)', '膳食纤维(g)',
                                  '营养健康指数NHI']].reset_index(drop=True))

    st.subheader("各类食物中最健康的 Top3")
    all_top3 = []
    for _, sub in df.groupby('食物类别'):
        all_top3.append(sub.nlargest(3, '营养健康指数NHI')[
            ['食物类别', '食物标准名称', '营养健康指数NHI']])
    st.dataframe(pd.concat(all_top3).reset_index(drop=True))

    st.subheader("各类食物平均 NHI 对比")
    cat_stats = df.groupby('食物类别')['营养健康指数NHI'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    cat_stats.plot(kind='barh', color='#55A868', ax=ax)
    ax.set_xlabel('平均 NHI')
    st.pyplot(fig); plt.close(fig)


# ---------- Tab 6: 食物查询 ----------
with tab6:
    st.header("🔍 单条食物查询")
    st.caption("输入食物名称关键词，查看其营养详情与 NHI")

    kw = st.text_input("关键词（英文）", value="tofu")
    matched = df[df['食物标准名称'].str.contains(kw, case=False, na=False)]

    if len(matched) == 0:
        st.warning("没有匹配的食物")
    else:
        pick = st.selectbox("选择食物", matched['食物标准名称'].tolist())
        row = df[df['食物标准名称'] == pick].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("NHI", f"{row['营养健康指数NHI']:.1f}")
        c2.metric("热量", f"{row['热量(kcal)']:.0f} kcal")
        c3.metric("类别", row['食物类别'])

        st.subheader("营养成分详情")
        info = pd.DataFrame({
            '指标': ['蛋白质(g)', '脂肪(g)', '碳水化合物(g)', '糖(g)',
                     '膳食纤维(g)', '钠(mg)', '钙(mg)', '铁(mg)', '维生素C(mg)'],
            '数值': [row[c] for c in ['蛋白质(g)', '脂肪(g)', '碳水化合物(g)',
                                     '糖(g)', '膳食纤维(g)', '钠(mg)',
                                     '钙(mg)', '铁(mg)', '维生素C(mg)']]
        })
        st.dataframe(info)

        fig, ax = plt.subplots(figsize=(8, 4))
        plt.barh(info['指标'], info['数值'], color='steelblue')
        plt.title(f'{pick} 营养组成')
        st.pyplot(fig); plt.close(fig)

st.divider()
st.caption("© 2026 食物营养成分分析平台 · 数据仅供参考，不作医学建议")