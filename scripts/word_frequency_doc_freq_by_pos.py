"""计算去除停用词后每个词的词频和文档频，并按词性分组、从高到低排列，同时包含 BERT 向量表示。

输入：
- `result/preprocessing/scenic_comments_tokenized_pos.xlsx`（由 `tokenize_and_pos_tag_comments.py` 生成）
- `word_embeddings_bert_with_pos.pkl`（由 `bert_embed_words_with_pos.py` 生成，包含已有的 BERT 向量）
输出：
- `result/preprocessing/word_frequency_doc_freq_by_pos.xlsx`（包含按词性分组的词频、文档频统计和向量信息）

注意：BERT 向量数据已保存在 `word_embeddings_bert_with_pos.pkl` 中，本脚本不再重复保存。
"""

import os
import ast
import pickle
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple

# 输出文件夹
OUTPUT_DIR = os.path.join("result", "preprocessing")

# 输入文件
INPUT_EXCEL = os.path.join(OUTPUT_DIR, "scenic_comments_tokenized_pos.xlsx")
INPUT_SHEET = "scenic_comments_tokenized_pos"

# 输入文件（已有的 BERT 向量）
INPUT_PKL = "word_embeddings_bert_with_pos.pkl"

# 输出文件
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "word_frequency_doc_freq_by_pos.xlsx")
OUTPUT_SHEET_FREQ = "词频文档频统计_按词性"
OUTPUT_SHEET_SUMMARY = "词性汇总"


def parse_list_column(value):
    """解析 Excel 中可能被读取为字符串的列表。"""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except:
            return []
    return []


def extract_word_pos_from_tags(pos_tags: List[Tuple[str, str]], clean_tokens_set: set) -> Dict[str, str]:
    """
    从 pos_tags 中提取每个词对应的词性，只保留在 clean_tokens 中出现的词。
    
    参数:
        pos_tags: 词性标注列表 [(词, 词性), ...]
        clean_tokens_set: clean_tokens 中所有词的集合
    
    返回:
        word_pos: 字典，{词: 最常见的词性}
    """
    word_pos_counter = Counter()
    
    for word, pos in pos_tags:
        if word in clean_tokens_set:  # 只保留在 clean_tokens 中的词
            word_pos_counter[(word, pos)] += 1
    
    # 为每个词选择最常见的词性
    word_pos = {}
    word_pos_temp = {}  # {词: Counter({词性: 出现次数})}
    
    for (word, pos), count in word_pos_counter.items():
        if word not in word_pos_temp:
            word_pos_temp[word] = Counter()
        word_pos_temp[word][pos] += count
    
    for word, pos_counter in word_pos_temp.items():
        most_common_pos = pos_counter.most_common(1)[0][0]
        word_pos[word] = most_common_pos
    
    return word_pos


def main():
    print("=" * 60)
    print("计算去除停用词后每个词的词频和文档频，并按词性分组、从高到低排列")
    print("输入文件:")
    print(f"  - {INPUT_EXCEL} (分词后的评论数据)")
    print(f"  - {INPUT_PKL} (已有的 BERT 向量数据)")
    print("输出文件:", OUTPUT_EXCEL)
    print("=" * 60)
    
    # 创建输出文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出文件夹: {OUTPUT_DIR}\n")
    
    # 1. 读取分词后的 Excel
    df = pd.read_excel(INPUT_EXCEL, sheet_name=INPUT_SHEET)
    required_cols = {"clean_tokens", "pos_tags"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"输入文件缺少必要列: {missing}")
    
    print(f"成功读取 {len(df)} 条评论")
    
    # 2. 解析 clean_tokens 和 pos_tags 列
    print("\n解析数据列...")
    df['clean_tokens_parsed'] = df['clean_tokens'].apply(parse_list_column)
    df['pos_tags_parsed'] = df['pos_tags'].apply(parse_list_column)
    
    # 3. 提取所有唯一的词（从 clean_tokens 中）
    print("提取所有唯一词...")
    all_unique_words = set()
    for tokens in df['clean_tokens_parsed']:
        all_unique_words.update(tokens)
    
    print(f"共有 {len(all_unique_words)} 个唯一词")
    
    # 4. 统计词频（从 clean_tokens 中）
    print("统计词频...")
    word_freq = Counter()
    for tokens in df['clean_tokens_parsed']:
        word_freq.update(tokens)
    
    print(f"统计到 {len(word_freq)} 个词的词频")
    
    # 4.5. 统计文档频（每个词出现在多少个不同的评论中）
    print("统计文档频...")
    word_doc_freq = {}  # {词: set(文档索引)}
    
    for idx, tokens in enumerate(df['clean_tokens_parsed']):
        for word in tokens:
            if word not in word_doc_freq:
                word_doc_freq[word] = set()
            word_doc_freq[word].add(idx)
    
    # 转换为文档频计数
    word_doc_freq_count = {word: len(doc_set) for word, doc_set in word_doc_freq.items()}
    print(f"统计到 {len(word_doc_freq_count)} 个词的文档频")
    
    # 5. 提取每个词的词性（从 pos_tags 中，只保留在 clean_tokens 中出现的词）
    print("提取词性信息...")
    word_pos_mapping = {}
    
    for idx, row in df.iterrows():
        pos_tags = row['pos_tags_parsed']
        clean_tokens = row['clean_tokens_parsed']
        clean_tokens_set = set(clean_tokens)
        
        # 从 pos_tags 中提取词性，只保留在 clean_tokens 中的词
        for word, pos in pos_tags:
            if word in clean_tokens_set:
                if word not in word_pos_mapping:
                    word_pos_mapping[word] = Counter()
                word_pos_mapping[word][pos] += 1
    
    # 为每个词选择最常见的词性
    word_pos = {}
    for word, pos_counter in word_pos_mapping.items():
        most_common_pos = pos_counter.most_common(1)[0][0]
        word_pos[word] = most_common_pos
    
    print(f"为 {len(word_pos)} 个词找到了词性信息")
    
    # 6. 按词性分组，词频从高到低排列
    print("\n按词性分组并排序...")
    words_by_pos = {}  # {词性: [(词, 词频), ...]}
    
    for word, freq in word_freq.items():
        pos = word_pos.get(word, "unknown")
        if pos not in words_by_pos:
            words_by_pos[pos] = []
        words_by_pos[pos].append((word, freq))
    
    # 对每个词性内的词按词频从高到低排序
    for pos in words_by_pos:
        words_by_pos[pos].sort(key=lambda x: x[1], reverse=True)
    
    # 7. 加载已有的 BERT 向量数据
    print("\n加载已有的 BERT 向量数据...")
    try:
        with open(INPUT_PKL, 'rb') as f:
            word_embeddings_with_pos_data = pickle.load(f)
        print(f"✓ 成功加载 {len(word_embeddings_with_pos_data)} 个词的 BERT 向量数据")
        
        # 提取向量字典和词性映射（从已有数据中）
        word_embeddings_dict = {}
        word_pos_from_vectors = {}
        
        for word, data in word_embeddings_with_pos_data.items():
            if isinstance(data, dict) and 'embedding' in data:
                word_embeddings_dict[word] = data['embedding']
                word_pos_from_vectors[word] = data.get('pos', 'unknown')
        
        print(f"  提取到 {len(word_embeddings_dict)} 个词的向量")
        if len(word_embeddings_dict) > 0:
            print(f"  向量维度: {list(word_embeddings_dict.values())[0].shape}")
        
        # 如果已有数据中有词性信息，优先使用（因为它是从 pos_tags 中准确提取的）
        if word_pos_from_vectors:
            print(f"  使用已有数据中的词性信息（覆盖从 pos_tags 提取的词性）")
            word_pos.update(word_pos_from_vectors)
            
    except FileNotFoundError:
        print(f"✗ 警告: 找不到 {INPUT_PKL} 文件")
        print("  请先运行 bert_embed_words_with_pos.py 生成 BERT 向量数据")
        print("  将只使用词频和词性信息，不包含向量数据")
        word_embeddings_dict = {}
    except Exception as e:
        print(f"✗ 加载向量数据时出错: {str(e)}")
        print("  将只使用词频和词性信息，不包含向量数据")
        word_embeddings_dict = {}
    
    # 8. 准备导出数据（包含向量信息）
    print("\n准备导出数据...")
    
    # 详细数据：所有词及其词频、文档频、词性、向量信息
    excel_data = []
    for pos in sorted(words_by_pos.keys()):
        for word, freq in words_by_pos[pos]:
            doc_freq = word_doc_freq_count.get(word, 0)
            embedding = word_embeddings_dict.get(word, None)
            if embedding is not None:
                excel_data.append({
                    '词性': pos,
                    '词': word,
                    '词频': freq,
                    '文档频': doc_freq,
                    '向量维度': embedding.shape[0],
                    '向量前5个值': str(embedding[:5].tolist())  # 只显示前5个值作为示例
                })
            else:
                excel_data.append({
                    '词性': pos,
                    '词': word,
                    '词频': freq,
                    '文档频': doc_freq,
                    '向量维度': 'N/A',
                    '向量前5个值': 'N/A'
                })
    
    # 汇总数据：每个词性的统计信息
    summary_data = []
    for pos in sorted(words_by_pos.keys()):
        words_list = words_by_pos[pos]
        total_words = len(words_list)
        total_freq = sum(freq for _, freq in words_list)
        avg_freq = total_freq / total_words if total_words > 0 else 0
        max_freq = words_list[0][1] if words_list else 0
        
        # 文档频统计
        doc_freqs = [word_doc_freq_count.get(word, 0) for word, _ in words_list]
        total_doc_freq = sum(doc_freqs)
        avg_doc_freq = total_doc_freq / total_words if total_words > 0 else 0
        max_doc_freq = max(doc_freqs) if doc_freqs else 0
        
        # Top10 高频词（显示词频和文档频）
        top10_words = ', '.join([
            f"{word}(词频:{freq},文档频:{word_doc_freq_count.get(word, 0)})" 
            for word, freq in words_list[:10]
        ])
        
        summary_data.append({
            '词性': pos,
            '词数量': total_words,
            '总词频': total_freq,
            '平均词频': round(avg_freq, 2),
            '最高词频': max_freq,
            '总文档频': total_doc_freq,
            '平均文档频': round(avg_doc_freq, 2),
            '最高文档频': max_doc_freq,
            'Top10高频词': top10_words
        })
    
    # 9. 导出到 Excel
    print("导出到 Excel...")
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        # 详细数据表
        df_freq = pd.DataFrame(excel_data)
        df_freq.to_excel(writer, index=False, sheet_name=OUTPUT_SHEET_FREQ)
        
        # 汇总表
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, index=False, sheet_name=OUTPUT_SHEET_SUMMARY)
    
    print(f"\n✓ Excel 文件已保存到 {OUTPUT_EXCEL}")
    print(f"  - Sheet1 ({OUTPUT_SHEET_FREQ}): 详细词频统计（按词性分组，词频从高到低，包含向量信息）")
    print(f"  - Sheet2 ({OUTPUT_SHEET_SUMMARY}): 词性汇总统计")
    
    # 10. 显示统计信息
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)
    print(f"\n总词数: {len(word_freq)}")
    print(f"总词频: {sum(word_freq.values())}")
    print(f"总文档频: {sum(word_doc_freq_count.values())}")
    print(f"\n词性分布:")
    for pos in sorted(words_by_pos.keys()):
        count = len(words_by_pos[pos])
        total_freq = sum(freq for _, freq in words_by_pos[pos])
        doc_freqs = [word_doc_freq_count.get(word, 0) for word, _ in words_by_pos[pos]]
        total_doc_freq = sum(doc_freqs)
        print(f"  {pos}: {count} 个词，总词频 {total_freq}，总文档频 {total_doc_freq}")
    
    print("\n各词性 Top5 高频词（词频/文档频）:")
    for pos in sorted(words_by_pos.keys()):
        top5 = words_by_pos[pos][:5]
        words_str = ', '.join([
            f"{word}(词频:{freq},文档频:{word_doc_freq_count.get(word, 0)})" 
            for word, freq in top5
        ])
        print(f"  {pos}: {words_str}")
    
    print("\n完成！")


if __name__ == "__main__":
    main()
