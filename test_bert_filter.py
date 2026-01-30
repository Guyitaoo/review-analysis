"""
测试BERT句子筛选器
"""
from bert_filter import BERTSentenceFilter, print_results, analyze_results
from test_data import vocabulary, sentences, expected_matches
from debug_utils import debug_vocab_vectors, debug_noun_similarities


if __name__ == "__main__":
    print("="*80)
    print("BERT句子筛选测试")
    print("="*80)
    print(f"\n词表 ({len(vocabulary)} 个词): {vocabulary}")
    print(f"待筛选句子数: {len(sentences)}")
    
    # 创建筛选器
    filter_tool = BERTSentenceFilter(
        vocabulary,
        model_name='bert-base-chinese'  # 首次运行会自动下载模型
    )
    
    # 显示词表向量调试信息
    debug_vocab_vectors(vocabulary, filter_tool.vocab_vectors)
    
    # 测试不同的阈值
    thresholds = [0.7]
    
    for threshold in thresholds:
        print(f"\n{'#'*80}")
        print(f"测试阈值: {threshold}")
        print(f"{'#'*80}")
        
        # 筛选句子
        results = filter_tool.filter_sentences(sentences, threshold=threshold)
        
        # 显示名词相似度调试信息（需要访问内部数据）
        # 注意：这需要在filter_sentences中添加返回相似度矩阵的功能
        # 暂时注释掉，如果需要可以修改filter_sentences方法
        
        # 打印结果
        print_results(results, threshold)
        
        # 分析结果
        analyze_results(results, sentences, expected_matches)
        
        print("\n")

