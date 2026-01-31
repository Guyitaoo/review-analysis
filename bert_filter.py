"""
基于BERT的句子筛选器
用于筛选出包含词表中词的近义词或同义词的句子
先对句子进行分词，提取名词，然后对名词与词表进行相似度计算
"""
# 忽略 jieba 的弃用警告
from typing import Any
import warnings
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

import warnings
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')

import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 
import jieba.posseg as pseg
import os


class BERTSentenceFilter:
    def __init__(self, vocabulary, model_name='bert-base-chinese', batch_size=32, stopwords_file='assets/hit_stopwords.txt'):
        # 初始化
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载停用词表
        self.stopwords = self._load_stopwords(stopwords_file)
        # 加载BERT模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        # 预计算词表的向量表示
        self.vocab_vectors = self._encode_batch(vocabulary)
    
    def _load_stopwords(self, stopwords_file):
        """加载停用词表"""
        stopwords = set()
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:  # 跳过空行
                    stopwords.add(word)
        return stopwords

    #  提取句子中的名词
    def _extract_nouns(self, sentence):
        words = pseg.cut(sentence)
        pure_noun_flags = ['n', 'nr', 'ns', 'nt', 'nz']
        nouns = []
        for word, flag in words:
            if flag in pure_noun_flags:
                word_clean = word.strip()
                if len(word_clean) > 0 and word_clean not in self.stopwords and len(word_clean) > 1:
                    nouns.append(word_clean)
        return nouns
    
    # 将文本编码为 BERT 向量
    def _encode_batch(self, texts):
        all_vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size] 
            inputs = self.tokenizer(
                batch_texts,         # 要处理的文本列表
                padding=True,        # 填充：短文本补零到相同长度
                truncation=True,     # 截断：超长文本截断
                max_length=128,      # 最大长度：128个token
                return_tensors='pt'  # 返回格式：PyTorch张量
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()} # 将张量移动到GPU
            with torch.no_grad(): # 禁用梯度计算，节省内存和计算资源
                outputs = self.model(**inputs) # BERT模型编码
                batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy() # 提取[CLS]标记的向量
                all_vectors.extend(batch_vectors) # 将批量向量添加到结果列表
        return np.array(all_vectors) 
    
    # 筛选出包含与词表意思相近或相同的句子
    def filter_sentences(self, sentences, threshold=0.5, return_all_matches=False): # 默认阈值为0.5
        # 提取句子中的名词
        sentence_nouns = []
        for sentence in sentences:
            nouns = self._extract_nouns(sentence)
            sentence_nouns.append({'sentence': sentence, 'nouns': nouns})
        # 收集所有唯一的名词
        all_nouns = []
        noun_to_sentences = {}
        for idx, item in enumerate[Any](sentence_nouns):
            for noun in item['nouns']:
                if noun not in noun_to_sentences:
                    all_nouns.append(noun)
                    noun_to_sentences[noun] = []
                noun_to_sentences[noun].append(idx)
        if not all_nouns:
            return []
        # 批量编码所有提取出的名词
        noun_vectors = self._encode_batch(all_nouns)
        # 计算名词与词表的相似度矩阵
        similarity_matrix = cosine_similarity(noun_vectors, self.vocab_vectors)
        # 对每个句子，找到其名词与词表的匹配
        results = []
        for idx, item in enumerate(sentence_nouns):
            sentence = item['sentence']
            nouns = item['nouns']
            if not nouns:
                continue
            max_similarity = 0
            best_match_word = None
            best_match_noun = None
            all_matches = []
            noun_to_matches = {}
            for noun in nouns:
                if noun in noun_to_sentences:
                    # 先检查直接匹配：如果名词包含词表中的关键词，直接匹配
                    direct_match = None
                    # 优先检查完全匹配
                    if noun in self.vocabulary:
                        direct_match = noun
                    else:
                        # 检查包含匹配：名词包含词表词，或词表词包含名词
                        matched_words = []
                        for word in self.vocabulary:
                            if word in noun or noun in word:
                                matched_words.append(word)
                        # 如果找到多个匹配，选择最长的词
                        if matched_words:
                            direct_match = max(matched_words, key=len)
                    # 如果找到直接匹配，优先使用直接匹配
                    if direct_match:
                        candidate_sim = 1.0  # 直接匹配的相似度设为1.0
                        candidate_word = direct_match
                        noun_matches = [{'word': direct_match, 'similarity': 1.0, 'noun': noun}]
                        # 更新最佳匹配
                        if candidate_sim > max_similarity:
                            max_similarity = candidate_sim
                            best_match_word = candidate_word
                            best_match_noun = noun
                        # 记录匹配信息
                        noun_to_matches[noun] = noun_matches
                        if return_all_matches:
                            all_matches.extend(noun_matches)
                        # 直接匹配后，跳过BERT相似度计算
                        continue
                    
                    # 如果没有直接匹配，使用BERT相似度匹配
                    noun_idx = all_nouns.index(noun)
                    noun_similarities = similarity_matrix[noun_idx]
                    sorted_indices = np.argsort(noun_similarities)[::-1]
                    sorted_sims = noun_similarities[sorted_indices]
                    candidate_sim = 0
                    candidate_word = None
                    noun_matches = []                   
                    # 遍历所有词表词，找到所有超过阈值的匹配
                    for rank in range(len(sorted_indices)):
                        word_idx = sorted_indices[rank]
                        word = self.vocabulary[word_idx]
                        sim = sorted_sims[rank]                       
                        if sim < threshold:
                            continue                        
                        # 检查是否应该保留这个匹配
                        should_keep = True
                        
                        # 对"电脑"使用特殊的匹配策略
                        if word == '电脑':
                            if len(sorted_sims) > rank + 1:
                                gap = sorted_sims[rank] - sorted_sims[rank + 1]
                                # 检查相似度差距：如果差距 < 0.03 且相似度 < 0.95，视为误匹配并跳过
                                if gap < 0.03 and sim < 0.95:
                                    should_keep = False
                                # 保留高置信度匹配：相似度 ≥ 0.95 或差距 ≥ 0.03
                                # (should_keep 已经默认为 True，所以这里不需要额外设置)
                        else:
                            # 非"电脑"词，使用通用过滤逻辑
                            if len(sorted_sims) > rank + 1:
                                gap = sorted_sims[rank] - sorted_sims[rank + 1]
                                # 如果差距非常小且相似度不够高，跳过
                                if gap < 0.01 and sim < 0.90:
                                    should_keep = False                       
                        if should_keep:
                            match_info = {'word': word, 'similarity': sim, 'noun': noun}
                            noun_matches.append(match_info)
                            if sim > candidate_sim:
                                candidate_sim = sim
                                candidate_word = word
                            if not return_all_matches:
                                break
                    if candidate_sim > max_similarity:
                        max_similarity = candidate_sim
                        best_match_word = candidate_word
                        best_match_noun = noun
                    if noun_matches:
                        noun_to_matches[noun] = noun_matches
                        if return_all_matches:
                            all_matches.extend(noun_matches)
            # 如果最大相似度达到阈值，添加到结果
            if max_similarity >= threshold:
                result = {
                    'sentence': sentence,
                    'similarity': float(max_similarity),
                    'matched_word': best_match_word,
                    'matched_noun': best_match_noun,
                    'all_nouns': nouns,
                    'index': idx
                }
                if return_all_matches:
                    # 去重：同一个词表词只保留相似度最高的
                    word_to_match = {}
                    for match in all_matches:
                        word = match['word']
                        if word not in word_to_match or match['similarity'] > word_to_match[word]['similarity']:
                            word_to_match[word] = match
                    result['matched_words'] = sorted(
                        word_to_match.values(),
                        key=lambda x: x['similarity'],
                        reverse=True
                    )
                    result['noun_matches'] = {
                        noun: sorted(matches, key=lambda x: x['similarity'], reverse=True)
                        for noun, matches in noun_to_matches.items()
                    }
                results.append(result)
        return results

if __name__ == "__main__":
    # 导入测试数据
    try:
        from test_data import vocabulary, sentences, expected_matches
    except ImportError:
        print("错误: 找不到 test_data.py 文件")
        print("请确保 test_data.py 文件存在，或者手动定义 vocabulary 和 sentences")
        exit(1)
    
    print("="*80)
    print("BERT句子筛选器 - 直接运行测试")
    print("="*80)
    print(f"\n词表 ({len(vocabulary)} 个词): {vocabulary}")
    print(f"待筛选句子数: {len(sentences)}")
    
    # 创建筛选器
    print("\n正在初始化BERT模型...")
    filter_tool = BERTSentenceFilter(
        vocabulary,
        model_name='bert-base-chinese'
    )
    
    # 设置阈值
    threshold = 0.7
    print(f"\n使用相似度阈值: {threshold}")
    print("-"*80)
    
    # 筛选句子
    results = filter_tool.filter_sentences(sentences, threshold=threshold, return_all_matches=False)
    
    # 打印结果
    print(f"\n找到 {len(results)} 个匹配的句子:\n")
    for i, result in enumerate(results, 1):
        print(f"[{i}] {result['sentence']}")
        print(f"     匹配词: {result['matched_word']} (相似度: {result['similarity']:.4f})")
        print(f"     匹配名词: {result['matched_noun']}")
        print(f"     提取的所有名词: {result['all_nouns']}")
        print()
    
    # 如果有预期结果，进行对比分析
    if expected_matches:
        print("\n" + "="*80)
        print("结果分析")
        print("="*80)
        
        matched_indices = {r['index'] for r in results}
        correct = 0
        total_expected = len([v for v in expected_matches.values() if v])
        false_positive = []  # 误匹配：不应该匹配但匹配了，或匹配错了
        false_negative = []  # 漏匹配：应该匹配但没匹配
        
        for i, sentence in enumerate(sentences):
            expected_word = expected_matches.get(sentence)
            matched_result = next((r for r in results if r['index'] == i), None)
            is_matched = matched_result is not None
            
            if expected_word:
                # 应该匹配的句子
                if is_matched and matched_result['matched_word'] == expected_word:
                    correct += 1
                elif is_matched:
                    # 匹配了但词不对
                    false_positive.append({
                        'sentence': sentence,
                        'expected_word': expected_word,
                        'matched_word': matched_result['matched_word'],
                        'matched_noun': matched_result.get('matched_noun'),
                        'similarity': matched_result['similarity'],
                        'all_nouns': matched_result.get('all_nouns', []),
                        'index': i
                    })
                else:
                    # 应该匹配但没匹配
                    false_negative.append({
                        'sentence': sentence,
                        'expected_word': expected_word,
                        'index': i
                    })
            else:
                # 不应该匹配的句子
                if is_matched:
                    false_positive.append({
                        'sentence': sentence,
                        'expected_word': None,
                        'matched_word': matched_result['matched_word'],
                        'matched_noun': matched_result.get('matched_noun'),
                        'similarity': matched_result['similarity'],
                        'all_nouns': matched_result.get('all_nouns', []),
                        'index': i
                    })
        
        if total_expected > 0:
            recall = correct / total_expected * 100
            print(f"正确匹配: {correct}/{total_expected}")
            print(f"召回率: {recall:.1f}%")
        
        if len(results) > 0:
            precision = correct / len(results) * 100
            print(f"精确率: {precision:.1f}%")
        
        # 输出误匹配详情
        if false_positive:
            print(f"\n{'='*80}")
            print(f"误匹配详情（共 {len(false_positive)} 个）")
            print(f"{'='*80}")
            for i, item in enumerate(false_positive, 1):
                print(f"\n[{i}] 句子: {item['sentence']}")
                if item['expected_word']:
                    print(f"     预期匹配词: {item['expected_word']}")
                    print(f"     实际匹配词: {item['matched_word']} (相似度: {item['similarity']:.4f})")
                else:
                    print(f"     预期: 不应该匹配")
                    print(f"     实际匹配词: {item['matched_word']} (相似度: {item['similarity']:.4f})")
                print(f"     匹配的名词: {item['matched_noun']}")
                print(f"     提取的所有名词: {item['all_nouns']}")
                print(f"     句子索引: {item['index']}")
        
        # 输出漏匹配详情
        if false_negative:
            print(f"\n{'='*80}")
            print(f"漏匹配详情（共 {len(false_negative)} 个）")
            print(f"{'='*80}")
            for i, item in enumerate(false_negative, 1):
                print(f"\n[{i}] 句子: {item['sentence']}")
                print(f"     预期匹配词: {item['expected_word']}")
                print(f"     句子索引: {item['index']}")
