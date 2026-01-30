"""
基于BERT的句子筛选器
用于筛选出包含词表中词的近义词或同义词的句子
先对句子进行分词，提取名词，然后对名词与词表进行相似度计算
"""
# 忽略 jieba 的弃用警告
import warnings
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')

import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 
import jieba.posseg as pseg


class BERTSentenceFilter:
    def __init__(self, vocabulary, model_name='bert-base-chinese', batch_size=32):

        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载BERT模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # 预计算词表的向量表示
        self.vocab_vectors = self._encode_batch(vocabulary)
    
    def _extract_nouns(self, sentence):
        """
        从句子中提取纯粹的名词
        
        Args:
            sentence: 输入句子
        
        Returns:
            名词列表
        """
        words = pseg.cut(sentence)
        pure_noun_flags = ['n', 'nr', 'ns', 'nt', 'nz']
        non_noun_blacklist = {
            '很', '的', '了', '在', '是', '有', '要', '会', '能', '可以',
            '换', '买', '用', '看', '做', '去', '来', '走', '跑', '吃',
            '喝', '睡', '想', '说', '听', '写', '读', '学', '工作',
            '有点', '一些', '这个', '那个', '什么', '怎么', '为什么'
        }
        
        nouns = []
        for word, flag in words:
            if flag in pure_noun_flags:
                word_clean = word.strip()
                if (len(word_clean) > 0 and 
                    word_clean not in non_noun_blacklist and
                    (len(word_clean) > 1 or word_clean in ['书', '猫', '狗'])):
                    nouns.append(word_clean)
        
        return nouns
    
    def _encode_batch(self, texts):
        """
        批量编码文本
        
        Args:
            texts: 文本列表
        
        Returns:
            文本向量数组
        """
        all_vectors = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_vectors.extend(batch_vectors)
        
        return np.array(all_vectors)
    
    def filter_sentences(self, sentences, threshold=0.5, return_all_matches=False):
        """
        筛选出包含与词表意思相近或相同的句子
        
        Args:
            sentences: 句子列表
            threshold: 相似度阈值，0-1之间
            return_all_matches: 如果为True，返回所有匹配的词表词
        
        Returns:
            匹配的句子列表，每个结果包含：
            - sentence: 句子文本
            - similarity: 最高相似度分数
            - matched_word: 匹配的词表词（相似度最高的）
            - matched_noun: 匹配到的名词
            - all_nouns: 句子中提取的所有名词
            - index: 句子在输入列表中的索引
            - matched_words: 所有匹配的词表词列表（仅当return_all_matches=True）
            - noun_matches: 每个名词对应的匹配信息字典（仅当return_all_matches=True）
        """
        # 提取句子中的名词
        sentence_nouns = []
        for sentence in sentences:
            nouns = self._extract_nouns(sentence)
            sentence_nouns.append({'sentence': sentence, 'nouns': nouns})
        
        # 收集所有唯一的名词
        all_nouns = []
        noun_to_sentences = {}
        for idx, item in enumerate(sentence_nouns):
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
        
        for i, sentence in enumerate(sentences):
            expected_word = expected_matches.get(sentence)
            if expected_word:
                matched_result = next((r for r in results if r['index'] == i), None)
                if matched_result and matched_result['matched_word'] == expected_word:
                    correct += 1
        
        if total_expected > 0:
            recall = correct / total_expected * 100
            print(f"正确匹配: {correct}/{total_expected}")
            print(f"召回率: {recall:.1f}%")
        
        if len(results) > 0:
            precision = correct / len(results) * 100
            print(f"精确率: {precision:.1f}%")
