"""从分词后的 Excel 读取评论，对 clean_tokens 中的词进行 BERT 向量表示，并保留词性信息。

输入：`result/preprocessing/scenic_comments_tokenized_pos.xlsx`（由 `tokenize_and_pos_tag_comments.py` 生成）
输出：
- `word_embeddings_bert_with_pos.pkl`：包含词、向量、词性的完整数据（用于后续聚类分析）

注意：不再输出Excel文件，因为result/preprocessing/word_frequency_doc_freq_by_pos.xlsx已包含词、词性、向量维度等所有相关信息。
"""

import os
import ast
import pandas as pd
import numpy as np
import pickle
import torch
import jieba.posseg as pseg
from transformers import AutoTokenizer, AutoModel
from collections import Counter
from typing import Dict, List, Tuple

# 输入文件
INPUT_EXCEL = os.path.join("result", "preprocessing", "scenic_comments_tokenized_pos.xlsx")
INPUT_SHEET = "scenic_comments_tokenized_pos"

# 输出文件
OUTPUT_PKL = "word_embeddings_bert_with_pos.pkl"
# 注意：不再输出Excel文件，因为word_frequency_doc_freq_by_pos.xlsx已包含所有相关信息


def get_word_embeddings_batch(words: List[str], model, tokenizer, device='cpu') -> Dict[str, np.ndarray]:
    """
    批量处理多个词，获取 BERT 向量表示。
    
    参数:
        words: 词列表
        model: BERT模型（应该已经在正确的device上）
        tokenizer: BERT分词器
        device: 设备
    
    返回:
        word_embeddings: 字典，{词: 向量}
    """
    word_embeddings = {}
    
    if not words:
        return word_embeddings
    
    # 过滤空词
    valid_words = [w for w in words if w and len(w.strip()) > 0]
    if not valid_words:
        return word_embeddings
    
    # 批量编码所有词
    inputs = tokenizer(
        valid_words,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # 获取所有词的hidden states: (batch_size, seq_len, hidden_size)
        hidden_states = outputs.last_hidden_state.cpu().numpy()
    
    # 为每个词提取向量
    for idx, word in enumerate(valid_words):
        # 获取该词对应的hidden states
        word_hidden = hidden_states[idx]  # (seq_len, hidden_size)
        
        # 排除[CLS]和[SEP]，对实际词的token求平均
        word_tokens = word_hidden[1:-1]  # 去掉[CLS]和[SEP]
        
        if len(word_tokens) > 0:
            word_embedding = np.mean(word_tokens, axis=0)
        else:
            # 如果只有[CLS]，就用它
            word_embedding = word_hidden[0]
        
        word_embeddings[word] = word_embedding
    
    return word_embeddings


def extract_word_pos_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """
    从 DataFrame 的 pos_tags 和 clean_tokens 列中提取每个词对应的最常见词性。
    逐行匹配，确保每个在 clean_tokens 中的词都能在对应的 pos_tags 中找到词性。
    
    参数:
        df: 包含 pos_tags 和 clean_tokens 列的 DataFrame
    
    返回:
        word_pos: 字典，{词: 最常见的词性}
    """
    word_pos_counter = {}  # {词: Counter({词性: 出现次数})}
    
    for idx, row in df.iterrows():
        # 解析 clean_tokens
        clean_tokens = row.get('clean_tokens', [])
        if pd.isna(clean_tokens):
            continue
        
        if isinstance(clean_tokens, str):
            try:
                clean_tokens = ast.literal_eval(clean_tokens)
            except:
                continue
        
        if not isinstance(clean_tokens, list):
            continue
        
        clean_tokens_set = set(clean_tokens)  # 当前行的 clean_tokens 集合
        
        # 解析 pos_tags
        pos_tags = row.get('pos_tags', [])
        if pd.isna(pos_tags):
            continue
        
        if isinstance(pos_tags, str):
            try:
                pos_tags = ast.literal_eval(pos_tags)
            except:
                continue
        
        if not isinstance(pos_tags, list):
            continue
        
        # 在同一行内匹配：只处理同时出现在 clean_tokens 和 pos_tags 中的词
        for word, pos in pos_tags:
            if word in clean_tokens_set:  # 只保留在当前行 clean_tokens 中的词
                if word not in word_pos_counter:
                    word_pos_counter[word] = Counter()
                word_pos_counter[word][pos] += 1
    
    # 为每个词选择最常见的词性
    word_pos = {}
    for word, pos_counter in word_pos_counter.items():
        if pos_counter:  # 确保有词性数据
            most_common_pos = pos_counter.most_common(1)[0][0]
            word_pos[word] = most_common_pos
    
    return word_pos


def main():
    print("=" * 60)
    print("从分词后的 Excel 读取评论，进行 BERT 向量表示（保留词性信息）")
    print("输入文件:", INPUT_EXCEL)
    print("输出文件:", OUTPUT_PKL)
    print("=" * 60)
    
    # 1. 读取分词后的 Excel
    df = pd.read_excel(INPUT_EXCEL, sheet_name=INPUT_SHEET)
    required_cols = {"clean_tokens", "pos_tags"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"输入文件缺少必要列: {missing}")
    
    print(f"成功读取 {len(df)} 条评论")
    
    # 2. 提取所有唯一的词（从 clean_tokens 中）
    print("\n开始提取所有唯一词...")
    all_unique_words = set()
    for tokens in df['clean_tokens']:
        if pd.isna(tokens):
            continue
        
        # Excel 读取时，列表可能被读取为字符串，需要解析
        if isinstance(tokens, str):
            try:
                tokens = ast.literal_eval(tokens)
            except:
                continue
        
        if not isinstance(tokens, list):
            continue
        
        all_unique_words.update(tokens)
    
    all_unique_words_list = list(all_unique_words)
    print(f"共有 {len(all_unique_words_list)} 个唯一词需要编码")
    
    # 3. 提取每个词对应的词性（逐行匹配 clean_tokens 和 pos_tags）
    print("\n提取每个词的词性信息...")
    word_pos_mapping = extract_word_pos_mapping(df)
    print(f"为 {len(word_pos_mapping)} 个词找到了词性信息")
    
    # 检查匹配情况，对未匹配的词使用jieba单独标注
    if len(word_pos_mapping) < len(all_unique_words):
        missing_words = all_unique_words - set(word_pos_mapping.keys())
        missing_count = len(missing_words)
        print(f"  有 {missing_count} 个词未在 pos_tags 中找到词性信息，尝试使用jieba单独标注...")
        
        # 对缺失的词使用jieba单独进行词性标注
        jieba_matched = 0
        for word in missing_words:
            try:
                # 使用jieba对单个词进行词性标注
                seg_result = list(pseg.cut(word))
                if seg_result:
                    # 取第一个结果的词性
                    _, pos = seg_result[0]
                    word_pos_mapping[word] = pos
                    jieba_matched += 1
            except:
                continue
        
        if jieba_matched > 0:
            print(f"  ✓ 使用jieba为 {jieba_matched} 个词补充了词性信息")
        
        # 最终检查
        final_missing = all_unique_words - set(word_pos_mapping.keys())
        if len(final_missing) > 0:
            print(f"  仍有 {len(final_missing)} 个词未找到词性信息，将标记为 'unknown'")
            if len(final_missing) <= 20:
                print(f"    未找到词性的词: {list(final_missing)}")
            else:
                print(f"    未找到词性的词（前20个）: {list(final_missing)[:20]}")
        else:
            print(f"  ✓ 所有词都已成功匹配到词性信息")
    else:
        print("✓ 所有词都成功匹配到词性信息")
    
    # 4. 检查 GPU 可用性
    print("\n" + "=" * 50)
    print("GPU/CPU 设备检查")
    print("=" * 50)
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✓ GPU可用！使用设备: {device}")
        print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
        batch_size = 64
    else:
        device = 'cpu'
        print(f"✗ GPU不可用，使用CPU")
        batch_size = 32
    print("=" * 50 + "\n")
    
    # 5. 加载 BERT 模型
    # 获取项目根目录（scripts的父目录）
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOCAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "model", "bert-base-chinese-local")
    print(f"正在从本地加载 BERT 模型: {LOCAL_MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    model = AutoModel.from_pretrained(LOCAL_MODEL_DIR)
    model = model.to(device)
    model.eval()
    print(f"模型已加载到 {device}\n")
    
    # 6. 使用批量方法对所有词进行 BERT 编码
    print(f"开始使用 BERT 对词进行编码（批量处理，批量大小: {batch_size}）...")
    word_embeddings_dict = {}
    
    total_batches = (len(all_unique_words_list) + batch_size - 1) // batch_size
    
    for i in range(0, len(all_unique_words_list), batch_size):
        batch_words = all_unique_words_list[i:i+batch_size]
        batch_embeddings = get_word_embeddings_batch(batch_words, model, tokenizer, device)
        word_embeddings_dict.update(batch_embeddings)
        
        current_batch = (i // batch_size) + 1
        if current_batch % 10 == 0 or current_batch == total_batches:
            print(f"已处理批次: {current_batch}/{total_batches} ({len(word_embeddings_dict)} 个词)")
    
    print(f"\n完成！共获得 {len(word_embeddings_dict)} 个词的 BERT 向量表示")
    if len(word_embeddings_dict) > 0:
        print(f"向量维度: {list(word_embeddings_dict.values())[0].shape}")
    
    # 7. 构建包含词性信息的完整数据结构
    print("\n构建包含词性信息的完整数据...")
    word_embeddings_with_pos = {}
    for word in word_embeddings_dict.keys():
        pos = word_pos_mapping.get(word, "unknown")  # 如果找不到词性，标记为 unknown
        word_embeddings_with_pos[word] = {
            'embedding': word_embeddings_dict[word],
            'pos': pos
        }
    
    print(f"共 {len(word_embeddings_with_pos)} 个词包含词性和向量信息")
    
    # 8. 保存为 pickle 文件（完整数据，包含向量）
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(word_embeddings_with_pos, f)
    print(f"\n✓ 完整数据已保存到 {OUTPUT_PKL}")
    print(f"  注意：不再输出Excel文件，因为word_frequency_doc_freq_by_pos.xlsx已包含所有相关信息")
    
    # 9. 显示统计信息
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)
    pos_counter = Counter([data['pos'] for data in word_embeddings_with_pos.values()])
    print("\n词性分布 Top10:")
    for pos, count in pos_counter.most_common(10):
        print(f"  {pos}: {count} 个词")
    
    print("\n完成！")


if __name__ == "__main__":
    main()
