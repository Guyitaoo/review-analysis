# 基于BERT的句子筛选测试

这个项目用于测试使用BERT模型筛选出包含词表中词的近义词或同义词的句子。

## 功能说明

- **分词和名词提取**: 使用jieba对句子进行分词和词性标注，提取出所有名词
- **BERT编码**: 使用BERT将词表和提取出的名词编码为向量
- **相似度计算**: 计算句子中的名词与词表中所有词的相似度
- **筛选匹配**: 根据阈值筛选出匹配的句子（句子中至少有一个名词与词表词相似度达到阈值）

## 工作流程

1. **分词和名词提取**: 对每个句子使用jieba进行词性标注，只提取纯粹的名词（n, nr, ns, nt, nz），不包括名动词（vn）、名形词（an）等
2. **去重**: 收集所有句子中提取出的唯一名词
3. **BERT编码**: 批量编码所有唯一名词和词表词为向量
4. **相似度计算**: 计算每个名词与词表所有词的余弦相似度
5. **匹配筛选**: 对每个句子，找到其名词与词表的最大相似度，如果达到阈值则保留

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```python
from bert_filter import BERTSentenceFilter

# 创建筛选器
filter_tool = BERTSentenceFilter(
    vocabulary=['手机', '电脑', '汽车'],  # 词表
    model_name='bert-base-chinese'
)

# 筛选句子
results = filter_tool.filter_sentences(
    sentences=['我的手机坏了', '这辆车很漂亮'],
    threshold=0.8
)
```

### 运行测试

```bash
python test_bert_filter.py
```

首次运行会自动下载BERT模型（bert-base-chinese），可能需要一些时间。

## 文件说明

- `test_data.py`: 包含测试用的词表和句子数据
- `bert_filter.py`: 主要的BERT筛选器实现（核心功能）
- `test_bert_filter.py`: 测试脚本
- `debug_utils.py`: 调试工具（用于分析词表向量和相似度）
- `requirements.txt`: 项目依赖

## 测试数据

- **词表**: 10个名词（手机、电脑、汽车、自行车、苹果、咖啡、猫、狗、书、电影）
- **句子**: 20个句子（16个应该匹配，4个可能不匹配）
- **测试重点**: 验证BERT对相似名词的识别能力（如同义词、近义词、相关词）

## 输出说明

程序会：
1. 加载BERT模型
2. 预计算词表向量
3. 对每个句子计算与词表的相似度
4. 根据阈值筛选结果
5. 显示匹配的句子、相似度分数和匹配的词
6. 分析结果准确率

## 调整参数

可以在创建 `BERTSentenceFilter` 时修改：
- `model_name`: BERT模型名称（默认'bert-base-chinese'）
- `batch_size`: 批处理大小（默认32）

在调用 `filter_sentences` 时修改：
- `threshold`: 相似度阈值（默认0.5，建议0.7-0.9）

## 调试工具

如果需要查看详细的调试信息，可以使用 `debug_utils.py` 中的函数：

```python
from debug_utils import debug_vocab_vectors, debug_noun_similarities

# 调试词表向量
debug_vocab_vectors(vocabulary, vocab_vectors)

# 调试名词相似度
debug_noun_similarities(vocabulary, all_nouns, similarity_matrix)
```
