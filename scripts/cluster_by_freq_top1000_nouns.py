"""基于词频最大的1000个名词进行BERT向量聚类和可视化。

输入：
- `result/preprocessing/word_frequency_doc_freq_by_pos.xlsx`（由 `word_frequency_doc_freq_by_pos.py` 生成）
- `word_embeddings_bert_with_pos.pkl`（由 `bert_embed_words_with_pos.py` 生成）
输出：
- `result/clustering/clustering_top1000_nouns_by_freq/` 文件夹下的所有结果文件
"""

import ast
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 输入文件
INPUT_EXCEL = os.path.join("result", "preprocessing", "word_frequency_doc_freq_by_pos.xlsx")
INPUT_SHEET = "词频文档频统计_按词性"
INPUT_PKL = "word_embeddings_bert_with_pos.pkl"

# 输出文件夹
OUTPUT_DIR = os.path.join("result", "clustering", "clustering_top1000_nouns_by_freq")

# 输出文件
OUTPUT_IMG = os.path.join(OUTPUT_DIR, "clustering_top1000_nouns_by_freq.png")
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "clustering_top1000_nouns_by_freq.xlsx")
OUTPUT_TXT = os.path.join(OUTPUT_DIR, "clustering_top1000_nouns_by_freq.txt")
OUTPUT_PKL_FILE = os.path.join(OUTPUT_DIR, "clustering_top1000_nouns_by_freq.pkl")

# 筛选参数
TOP_N_NOUNS = 1000  # 词频最大的1000个名词

# 聚类参数（可选：如果设置为非None，将使用固定值，否则自动选择最优k）
FIXED_N_CLUSTERS = None  # 例如：设置为 19 可以固定使用19个簇
MIN_CLUSTERS = 5  # 最小聚类数量（自动选择时不会低于此值，除非样本数太少）
MAX_CLUSTERS = 30  # 最大聚类数量

# 评价指标说明：
# 1. 轮廓系数（Silhouette Score）：范围 -1 到 1，值越大越好
#    - 衡量样本与其所属簇的相似度，以及与其他簇的差异度
#    - 问题：k值越大，轮廓系数可能越高（簇更小更紧密），但不一定最优
# 2. 肘部法则（Elbow Method）：基于惯性（inertia）的变化率
#    - 寻找惯性下降速度突然变慢的"肘部"点
# 3. 选择策略：优先选择轮廓系数高且k值适中的方案


def find_optimal_k(embeddings, max_k=None, min_k=None):
    """
    使用轮廓系数和肘部法则自动选择最优k值
    改进策略：避免选择过小的k值，优先考虑中等大小的k值
    """
    n_samples = len(embeddings)
    
    if n_samples < 10:
        return max(2, n_samples // 3)
    
    # 使用全局参数或默认值
    if max_k is None:
        max_k = min(MAX_CLUSTERS, int(n_samples * 0.8), 30)
    if min_k is None:
        min_k = max(MIN_CLUSTERS, 3)
    
    # 确保min_k不超过max_k
    if max_k < min_k:
        max_k = min_k + 5
    
    k_range = range(min_k, max_k + 1)
    silhouette_scores = []
    inertias = []
    
    print(f"  样本数量: {n_samples}")
    print(f"  尝试k值范围: {min_k} 到 {max_k} (共 {len(k_range)} 个值)")
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(embeddings)
        inertias.append(kmeans.inertia_)
        
        if len(set(labels)) > 1:
            sil_score = silhouette_score(embeddings, labels)
            silhouette_scores.append(sil_score)
        else:
            silhouette_scores.append(-1)
        
        print(f"    k={k:2d}: 轮廓系数={silhouette_scores[-1]:.4f}, 惯性={inertias[-1]:.2f}")
    
    if silhouette_scores and max(silhouette_scores) > 0:
        sorted_indices = np.argsort(silhouette_scores)[::-1]
        top_k_indices = sorted_indices[:min(10, len(sorted_indices))]  # 查看Top10
        top_k_values = [k_range[i] for i in top_k_indices]
        top_scores = [silhouette_scores[i] for i in top_k_indices]
        
        print(f"\n  轮廓系数Top10:")
        for i, (k_val, score) in enumerate(zip(top_k_values, top_scores), 1):
            print(f"    {i:2d}. k={k_val:2d}: {score:.4f}")
        
        # 改进的选择策略：平衡轮廓系数和簇数量
        # 1. 计算轮廓系数的增益（相对于前一个k值的提升）
        # 2. 选择轮廓系数较高但k值适中的方案
        # 3. 避免选择k值过大但轮廓系数提升很小的方案
        
        best_k = top_k_values[0]
        best_score = top_scores[0]
        
        # 如果最优k值过大（>20），检查是否有k值较小但轮廓系数相近的方案
        if best_k > 20:
            print(f"\n  警告：轮廓系数最高的k={best_k}可能过大，寻找更平衡的方案...")
            # 在Top10中寻找k值在8-20之间且轮廓系数可接受的方案
            balanced_candidates = [(k, s) for k, s in zip(top_k_values, top_scores) 
                                 if 8 <= k <= 20 and (best_score - s) < 0.03]
            if balanced_candidates:
                # 选择k值中等（接近15）且轮廓系数可接受的方案
                balanced_candidates.sort(key=lambda x: (abs(x[0] - 15), -x[1]))
                balanced_k, balanced_score = balanced_candidates[0]
                print(f"  → 找到更平衡的方案: k={balanced_k} (轮廓系数: {balanced_score:.4f}, "
                      f"与最优值差距: {best_score - balanced_score:.4f})")
                best_k = balanced_k
                best_score = balanced_score
            else:
                # 如果没有合适的平衡方案，选择Top10中k值最小的（但轮廓系数不能太差）
                for k_val, score in zip(top_k_values, top_scores):
                    if k_val <= 20 and (best_score - score) < 0.05:
                        print(f"  → 选择k={k_val} (轮廓系数: {score:.4f}, 与最优值差距: {best_score - score:.4f})")
                        best_k = k_val
                        best_score = score
                        break
        
        # 如果最优k值太小（<8），检查是否有k值稍大但轮廓系数相近的方案
        elif best_k < 8 and len(top_k_values) > 1:
            # 检查是否有k值在8-15之间且轮廓系数相近的方案
            for k_val, score in zip(top_k_values[1:], top_scores[1:]):
                if 8 <= k_val <= 15 and (best_score - score) < 0.02:
                    print(f"  → k={best_k} 的轮廓系数({best_score:.4f})与k={k_val}({score:.4f})相近，选择适中的k值: {k_val}")
                    best_k = k_val
                    best_score = score
                    break
        
        print(f"\n  → 最终选择k值: {best_k} (轮廓系数: {best_score:.4f})")
        print(f"  评价指标说明:")
        print(f"    - 轮廓系数范围: -1 到 1，值越大越好")
        print(f"    - 当前选择: 在轮廓系数和簇数量之间取得平衡")
        return best_k
    else:
        # 使用肘部法则
        if len(inertias) > 1:
            inertia_diffs = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
            if inertia_diffs:
                elbow_idx = np.argmax(inertia_diffs)
                optimal_k = k_range[elbow_idx + 1]
                print(f"\n  → 使用肘部法则，选择k值: {optimal_k}")
                return optimal_k
        
        # 默认估算：基于样本数，但至少为MIN_CLUSTERS
        optimal_k = max(MIN_CLUSTERS, min(15, n_samples // 20))
        print(f"\n  → 使用默认估算，选择k值: {optimal_k}")
        return optimal_k


def main():
    print("=" * 60)
    print("基于词频最大的1000个名词进行BERT向量聚类和可视化")
    print("=" * 60)
    
    # 创建输出文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n输出文件夹: {OUTPUT_DIR}")
    
    # 1. 读取词频统计Excel
    print("\n1. 读取词频统计数据...")
    df = pd.read_excel(INPUT_EXCEL, sheet_name=INPUT_SHEET)
    print(f"✓ 成功加载 {len(df)} 条词频数据")
    
    # 2. 筛选名词（词性以'n'开头）
    print("\n2. 筛选名词...")
    df_nouns = df[df['词性'].astype(str).str.startswith('n')].copy()
    print(f"✓ 找到 {len(df_nouns)} 个名词")
    
    # 3. 按词频排序，取前1000个
    print(f"\n3. 按词频排序，取前 {TOP_N_NOUNS} 个...")
    df_nouns_sorted = df_nouns.sort_values('词频', ascending=False)
    df_top_nouns = df_nouns_sorted.head(TOP_N_NOUNS)
    print(f"✓ 筛选出词频最大的 {len(df_top_nouns)} 个名词")
    
    # 显示Top20
    print("\n  词频Top20:")
    for i, row in df_top_nouns.head(20).iterrows():
        print(f"    {i+1:2d}. {row['词']}: 词频={row['词频']}, 文档频={row['文档频']}")
    
    # 4. 加载BERT向量
    print("\n4. 加载BERT向量...")
    try:
        with open(INPUT_PKL, 'rb') as f:
            word_embeddings_with_pos = pickle.load(f)
        print(f"✓ 成功加载 {len(word_embeddings_with_pos)} 个词的向量")
    except FileNotFoundError:
        print(f"✗ 错误: 找不到 {INPUT_PKL} 文件")
        exit(1)
    
    # 5. 提取筛选出的名词的向量
    print("\n5. 提取筛选出的名词的向量...")
    top_nouns_list = df_top_nouns['词'].tolist()
    word_embeddings_dict = {}
    word_freq_dict = {}
    word_doc_freq_dict = {}
    
    for word in top_nouns_list:
        if word in word_embeddings_with_pos:
            data = word_embeddings_with_pos[word]
            if isinstance(data, dict) and 'embedding' in data:
                word_embeddings_dict[word] = data['embedding']
                # 获取词频和文档频
                row = df_top_nouns[df_top_nouns['词'] == word].iloc[0]
                word_freq_dict[word] = row['词频']
                word_doc_freq_dict[word] = row['文档频']
    
    print(f"✓ 找到 {len(word_embeddings_dict)} 个词的向量")
    
    if len(word_embeddings_dict) == 0:
        print("✗ 错误: 没有找到任何词的向量")
        exit(1)
    
    # 6. 准备聚类数据
    print("\n6. 准备聚类数据...")
    words = list(word_embeddings_dict.keys())
    embeddings = np.array([word_embeddings_dict[word] for word in words])
    print(f"  词数量: {len(words)}")
    print(f"  向量维度: {embeddings.shape}")
    
    # 标准化
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # 7. 确定聚类数量
    print("\n7. 确定聚类数量...")
    if FIXED_N_CLUSTERS is not None:
        n_clusters = FIXED_N_CLUSTERS
        print(f"  使用固定聚类数量: {n_clusters}")
    else:
        n_clusters = find_optimal_k(embeddings_scaled)
        print(f"\n  最终使用聚类数量: {n_clusters}")
    
    # 8. 执行K-means聚类
    print("\n8. 执行K-means聚类...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_scaled)
    print(f"✓ 聚类完成，共 {n_clusters} 个簇")
    
    # 统计每个簇的词数量
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    print("\n各簇词数量统计:")
    for label, count in zip(unique_labels, counts):
        print(f"  簇 {label}: {count} 个词")
    
    # 9. 使用t-SNE降维到2D
    print("\n9. 使用t-SNE降维到2D...")
    print("  (这可能需要一些时间，请耐心等待...)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(words)-1), max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings_scaled)
    print("✓ 降维完成")
    
    # 10. 可视化聚类结果
    print("\n10. 生成可视化图表...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    cmap = plt.cm.get_cmap('tab20')
    colors = [cmap(i / n_clusters) for i in range(n_clusters)]
    
    # 左图：聚类散点图 + 簇边界
    ax1 = axes[0]
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_points = embeddings_2d[cluster_mask]
        
        if len(cluster_points) >= 3:
            try:
                hull = ConvexHull(cluster_points)
                for simplex in hull.simplices:
                    ax1.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 
                            color=colors[cluster_id], linewidth=2, alpha=0.3, linestyle='--')
            except:
                pass
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_points = embeddings_2d[cluster_mask]
        ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=[colors[cluster_id]], cmap='tab20', 
                   alpha=0.7, s=60, edgecolors='black', linewidths=0.5,
                   label=f'簇 {cluster_id}')
    
    ax1.set_title(f'BERT名词向量聚类结果（词频Top{TOP_N_NOUNS}） (K-means, k={n_clusters})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE 维度 1', fontsize=12)
    ax1.set_ylabel('t-SNE 维度 2', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    
    # 右图：标注代表词 + 簇边界
    ax2 = axes[1]
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_points = embeddings_2d[cluster_mask]
        
        if len(cluster_points) >= 3:
            try:
                hull = ConvexHull(cluster_points)
                for simplex in hull.simplices:
                    ax2.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 
                            color=colors[cluster_id], linewidth=2, alpha=0.2, linestyle='--')
            except:
                pass
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_points = embeddings_2d[cluster_mask]
        ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=[colors[cluster_id]], cmap='tab20', 
                   alpha=0.5, s=40, edgecolors='black', linewidths=0.3)
    
    # 标注代表词
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_embeddings = embeddings_scaled[cluster_mask]
        cluster_words = [words[i] for i in range(len(words)) if cluster_labels[i] == cluster_id]
        
        cluster_center = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
        top_indices = np.argsort(distances)[:min(5, len(cluster_words))]
        
        for idx in top_indices:
            word_idx = [i for i in range(len(words)) if cluster_labels[i] == cluster_id][idx]
            word = words[word_idx]
            x, y = embeddings_2d[word_idx]
            ax2.annotate(word, (x, y), fontsize=8, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[cluster_id], 
                                 alpha=0.8, edgecolor='black', linewidth=0.5),
                        ha='center', va='center')
    
    ax2.set_title('聚类结果（标注代表词，同簇用边界圈出）', fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE 维度 1', fontsize=12)
    ax2.set_ylabel('t-SNE 维度 2', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存为 {OUTPUT_IMG}")
    plt.close()
    
    # 11. 显示每个簇的详细词列表
    print("\n11. 各簇详细词列表:")
    print("=" * 60)
    for cluster_id in range(n_clusters):
        cluster_words = [words[i] for i in range(len(words)) if cluster_labels[i] == cluster_id]
        print(f"\n簇 {cluster_id} (共 {len(cluster_words)} 个词):")
        for i in range(0, len(cluster_words), 10):
            print("  " + "  ".join(cluster_words[i:i+10]))
    
    # 12. 保存聚类结果
    print("\n12. 保存聚类结果...")
    cluster_results = {
        'words': words,
        'embeddings_2d': embeddings_2d,
        'cluster_labels': cluster_labels,
        'n_clusters': n_clusters,
        'cluster_centers': kmeans.cluster_centers_,
        'word_frequencies': word_freq_dict,
        'word_doc_frequencies': word_doc_freq_dict
    }
    
    with open(OUTPUT_PKL_FILE, 'wb') as f:
        pickle.dump(cluster_results, f)
    print(f"✓ 聚类结果已保存到 {OUTPUT_PKL_FILE}")
    
    # 13. 导出为文本文件
    print("\n13. 导出聚类结果到文本文件...")
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"BERT名词向量聚类结果（词频Top{TOP_N_NOUNS}） - 按类别整理\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"聚类数量: {n_clusters}\n")
        f.write(f"总词数: {len(words)}\n")
        f.write(f"筛选条件: 词频最大的 {TOP_N_NOUNS} 个名词\n")
        f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n" + "=" * 60 + "\n\n")
        
        for cluster_id in range(n_clusters):
            cluster_words = [words[i] for i in range(len(words)) if cluster_labels[i] == cluster_id]
            cluster_words_with_freq = [(w, word_freq_dict.get(w, 0), word_doc_freq_dict.get(w, 0)) 
                                       for w in cluster_words]
            cluster_words_with_freq.sort(key=lambda x: x[1], reverse=True)
            
            f.write(f"【类别 {cluster_id}】\n")
            f.write(f"词数量: {len(cluster_words)}\n")
            f.write("-" * 60 + "\n")
            f.write("按词频排序:\n")
            for i, (word, freq, doc_freq) in enumerate(cluster_words_with_freq, 1):
                f.write(f"  {i:3d}. {word:15s} (词频: {freq:4d}, 文档频: {doc_freq:4d})\n")
            f.write("\n" + "=" * 60 + "\n\n")
    
    print(f"✓ 文本文件已保存: {OUTPUT_TXT}")
    
    # 14. 导出为Excel文件
    print("\n14. 导出聚类结果到Excel文件...")
    try:
        excel_data = []
        for cluster_id in range(n_clusters):
            cluster_words = [words[i] for i in range(len(words)) if cluster_labels[i] == cluster_id]
            cluster_words_with_freq = [(w, word_freq_dict.get(w, 0), word_doc_freq_dict.get(w, 0)) 
                                       for w in cluster_words]
            cluster_words_with_freq.sort(key=lambda x: x[1], reverse=True)
            
            for word, freq, doc_freq in cluster_words_with_freq:
                excel_data.append({
                    '类别': cluster_id,
                    '词': word,
                    '词频': freq,
                    '文档频': doc_freq,
                    '类别词数': len(cluster_words)
                })
        
        summary_data = []
        for cluster_id in range(n_clusters):
            cluster_words = [words[i] for i in range(len(words)) if cluster_labels[i] == cluster_id]
            cluster_words_with_freq = [(w, word_freq_dict.get(w, 0), word_doc_freq_dict.get(w, 0)) 
                                       for w in cluster_words]
            cluster_words_with_freq.sort(key=lambda x: x[1], reverse=True)
            
            top_words = ', '.join([f"{w}(词频:{freq},文档频:{doc_freq})" 
                                  for w, freq, doc_freq in cluster_words_with_freq[:10]])
            
            summary_data.append({
                '类别': cluster_id,
                '词数量': len(cluster_words),
                '平均词频': round(np.mean([word_freq_dict.get(w, 0) for w in cluster_words]), 2) if cluster_words else 0,
                '最高词频': max([word_freq_dict.get(w, 0) for w in cluster_words]) if cluster_words else 0,
                '平均文档频': round(np.mean([word_doc_freq_dict.get(w, 0) for w in cluster_words]), 2) if cluster_words else 0,
                '最高文档频': max([word_doc_freq_dict.get(w, 0) for w in cluster_words]) if cluster_words else 0,
                '前10个高频词': top_words
            })
        
        with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
            df_export = pd.DataFrame(excel_data)
            df_export.to_excel(writer, index=False, sheet_name='聚类结果')
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, index=False, sheet_name='类别汇总')
        
        print(f"✓ Excel文件已保存: {OUTPUT_EXCEL}")
        print(f"  - Sheet1: 聚类结果（详细数据）")
        print(f"  - Sheet2: 类别汇总（统计信息）")
        
    except Exception as e:
        print(f"✗ Excel导出失败: {str(e)}")
        print("  仅保存了文本文件")
    
    print("\n" + "=" * 60)
    print("聚类与可视化完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print(f"  1. {OUTPUT_IMG} - 聚类可视化图表")
    print(f"  2. {OUTPUT_PKL_FILE} - 完整聚类数据（二进制）")
    print(f"  3. {OUTPUT_TXT} - 聚类结果文本文件")
    print(f"  4. {OUTPUT_EXCEL} - 聚类结果Excel文件")


if __name__ == "__main__":
    main()
