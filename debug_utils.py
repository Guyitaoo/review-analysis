"""
调试工具：用于分析BERT句子筛选器的调试信息
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def debug_vocab_vectors(vocabulary, vocab_vectors):
    """
    调试信息：检查词表向量的特性
    
    Args:
        vocabulary: 词表列表
        vocab_vectors: 词表向量数组
    """
    print("\n" + "="*80)
    print("词表向量调试信息")
    print("="*80)
    
    # 1. 检查向量范数
    vector_norms = np.linalg.norm(vocab_vectors, axis=1)
    print("\n1. 词表向量范数:")
    for word, norm in zip(vocabulary, vector_norms):
        print(f"   {word}: {norm:.4f}")
    
    # 2. 计算词表词之间的相似度矩阵
    vocab_similarity_matrix = cosine_similarity(vocab_vectors, vocab_vectors)
    print("\n2. 词表词之间的相似度矩阵:")
    print("   ", end="")
    for word in vocabulary:
        print(f"{word:8s}", end="")
    print()
    for i, word1 in enumerate(vocabulary):
        print(f"   {word1:8s}", end="")
        for j, word2 in enumerate(vocabulary):
            if i == j:
                print(f"{1.0000:8.4f}", end="")
            else:
                print(f"{vocab_similarity_matrix[i][j]:8.4f}", end="")
        print()
    
    # 3. 检查每个词与其他词的平均相似度
    print("\n3. 每个词与其他词表的平均相似度:")
    for i, word in enumerate(vocabulary):
        # 排除自己
        other_similarities = np.delete(vocab_similarity_matrix[i], i)
        avg_sim = other_similarities.mean()
        max_sim = other_similarities.max()
        min_sim = other_similarities.min()
        print(f"   {word}: 平均={avg_sim:.4f}, 最大={max_sim:.4f}, 最小={min_sim:.4f}")
    
    # 4. 特别检查"电脑"
    if '电脑' in vocabulary:
        computer_idx = vocabulary.index('电脑')
        print(f"\n4. '电脑'的详细分析:")
        print(f"   向量范数: {vector_norms[computer_idx]:.4f}")
        print(f"   与其他词表的相似度:")
        for j, word2 in enumerate(vocabulary):
            if j != computer_idx:
                print(f"     {word2}: {vocab_similarity_matrix[computer_idx][j]:.4f}")
        other_similarities = np.delete(vocab_similarity_matrix[computer_idx], computer_idx)
        print(f"   平均相似度: {other_similarities.mean():.4f}")
        # 找到最大相似度对应的词
        max_sim_idx_in_others = np.argmax(other_similarities)
        # 需要映射回原始索引（排除computer_idx）
        other_indices = [j for j in range(len(vocabulary)) if j != computer_idx]
        max_sim_word_idx = other_indices[max_sim_idx_in_others]
        print(f"   最大相似度: {other_similarities.max():.4f} (与 {vocabulary[max_sim_word_idx]})")
    
    print("="*80 + "\n")


def debug_noun_similarities(vocabulary, all_nouns, similarity_matrix):
    """
    调试信息：显示名词与词表的相似度分布
    
    Args:
        vocabulary: 词表列表
        all_nouns: 所有提取出的名词列表
        similarity_matrix: 相似度矩阵 [名词数量 × 词表词数量]
    """
    print("\n" + "="*80)
    print("名词与词表相似度调试信息")
    print("="*80)
    
    # 统计每个词表词被匹配的次数（相似度最高的）
    vocab_match_count = {word: 0 for word in vocabulary}
    vocab_max_similarities = {word: [] for word in vocabulary}
    
    for i, noun in enumerate(all_nouns):
        noun_similarities = similarity_matrix[i]
        max_sim = np.max(noun_similarities)
        best_word_idx = np.argmax(noun_similarities)
        best_word = vocabulary[best_word_idx]
        
        vocab_match_count[best_word] += 1
        vocab_max_similarities[best_word].append((noun, max_sim))
    
    print("\n1. 每个词表词被匹配的次数（作为最高相似度）:")
    for word in vocabulary:
        print(f"   {word}: {vocab_match_count[word]} 次")
    
    # 显示"电脑"的匹配详情
    if '电脑' in vocabulary:
        computer_idx = vocabulary.index('电脑')
        print(f"\n2. '电脑'的匹配详情（前20个）:")
        computer_similarities = similarity_matrix[:, computer_idx]
        # 找到与"电脑"相似度最高的名词
        top_indices = np.argsort(computer_similarities)[::-1][:20]
        for rank, idx in enumerate(top_indices, 1):
            noun = all_nouns[idx]
            sim = computer_similarities[idx]
            # 检查这个名词的最高相似度词是什么
            max_sim_for_noun = np.max(similarity_matrix[idx])
            best_word_for_noun = vocabulary[np.argmax(similarity_matrix[idx])]
            marker = " ← 最高" if best_word_for_noun == '电脑' else ""
            print(f"   {rank:2d}. {noun:10s} 相似度={sim:.4f} (该名词最高匹配: {best_word_for_noun} {max_sim_for_noun:.4f}){marker}")
    
    # 统计误匹配到"电脑"的情况
    if '电脑' in vocabulary:
        computer_idx = vocabulary.index('电脑')
        false_matches = []
        for i, noun in enumerate(all_nouns):
            noun_similarities = similarity_matrix[i]
            max_sim = np.max(noun_similarities)
            best_word_idx = np.argmax(noun_similarities)
            if best_word_idx == computer_idx:
                false_matches.append((noun, max_sim))
        
        if false_matches:
            print(f"\n3. 匹配到'电脑'的名词（共{len(false_matches)}个）:")
            false_matches_sorted = sorted(false_matches, key=lambda x: x[1], reverse=True)
            for noun, sim in false_matches_sorted[:15]:  # 只显示前15个
                print(f"   {noun:10s} 相似度={sim:.4f}")
    
    print("="*80 + "\n")

