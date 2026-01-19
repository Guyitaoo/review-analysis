# 文件说明

## 快速开始

**推荐方式：使用主程序一键执行所有步骤**

```bash
python main.py
```

主程序会按照以下顺序自动执行所有步骤。

## 手动执行（可选）

如果需要单独执行某个步骤，可以进入 `scripts/` 文件夹运行对应的脚本。

---

## 开发环境验证

- `scripts/check_gpu.py`: 检测GPU是否可用

## 预加载

- `scripts/download_bert_model.py`: 从hugging face上下载BERT模型到本地

## 预处理

- `scripts/clean_comments_to_excel.py`: 对原始数据进行清洗+去重之后导出到 `result/preprocessing/` 文件夹

- `scripts/tokenize_and_pos_tag_comments.py`: 从预处理后的Excel读取评论，分词、去停用词、词性标注和名词抽取，输出到 `result/preprocessing/` 文件夹

- `scripts/bert_embed_words_with_pos.py`: 从分词后的Excel读取评论，对去停用词后词进行BERT向量表示，并保留词性信息。输出为PKL文件（用于后续聚类分析）。

- `scripts/word_frequency_doc_freq_by_pos.py`: 计算去除停用词后每个词的词频和文档频，并按词性分组、从高到低排列，同时包含 BERT 向量维度信息（完整向量保存在PKL文件中），输出到 `result/preprocessing/` 文件夹

## 聚类分析

- `scripts/cluster_by_freq_top1000_nouns.py`: 按照词频做聚类，输出到 `result/clustering/clustering_top1000_nouns_by_freq/` 文件夹

- `scripts/cluster_by_doc_freq_top500.py`: 按照文档频做聚类，输出到 `result/clustering/clustering_top500_by_doc_freq/` 文件夹

