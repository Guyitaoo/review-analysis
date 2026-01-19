"""从预处理后的 Excel 读取评论，完成分词、去停用词、词性标注和名词抽取。

对应原 `read_file.py` 中的步骤：
- 添加自定义词典
- 加载停用词表
- 分词得到 `tokens`
- 过滤得到 `clean_tokens`
- 词性标注 `pos_tags`
- 从词性标注中抽取名词 `nouns`

输入：`result/preprocessing/scenic_comments_cleaned_dedup.xlsx`（由 `clean_comments_to_excel.py` 生成）
输出：`result/preprocessing/scenic_comments_tokenized_pos.xlsx`
"""

import os
import pandas as pd
import jieba
import jieba.posseg as pseg
from typing import List, Tuple, Set

# 输出文件夹
OUTPUT_DIR = os.path.join("result", "preprocessing")

# 预处理后的 Excel 文件
INPUT_EXCEL = os.path.join(OUTPUT_DIR, "scenic_comments_cleaned_dedup.xlsx")
INPUT_SHEET = "scenic_comments_cleaned"

# 输出文件（增加分词与词性标注等列）
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "scenic_comments_tokenized_pos.xlsx")
OUTPUT_SHEET = "scenic_comments_tokenized_pos"

STOPWORDS_PATH = os.path.join("assets", "hit_stopwords.txt")


def load_stopwords(path: str) -> Set[str]:
    """加载停用词表，对应原 `read_file.py` 中 `load_stopwords` 逻辑。"""
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def filter_tokens(tokens: List[str], stopwords: Set[str]) -> List[str]:
    """去除停用词和空/单字等无效词，对应原 `filter_tokens`。"""
    return [
        w
        for w in tokens
        if w
        and w.strip()  # 去掉空字符串
        and w not in stopwords  # 去掉停用词
        and len(w) > 1  # 去掉单字（可按需要调整）
    ]


def pos_tag(text: str) -> List[Tuple[str, str]]:
    """对文本做词性标注，返回 (词, 词性) 列表。"""
    return [(w, flag) for w, flag in pseg.lcut(text)]


def extract_nouns(pos_tags: List[Tuple[str, str]]) -> List[str]:
    """从词性标注结果中提取所有名词（词性以 'n' 开头）。"""
    return [w for w, flag in pos_tags if flag.startswith("n")]


def main():
    print("=" * 60)
    print("从预处理后的 Excel 读取评论，进行分词 / 去停用词 / 词性标注 / 名词抽取")
    print("输入文件:", INPUT_EXCEL)
    print("输出文件:", OUTPUT_EXCEL)
    print("=" * 60)
    
    # 创建输出文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出文件夹: {OUTPUT_DIR}")

    # 1. 读取预处理后的 Excel
    df = pd.read_excel(INPUT_EXCEL, sheet_name=INPUT_SHEET)
    # 期望列：username, rating, content_clean
    required_cols = {"username", "rating", "content_clean"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"输入文件缺少必要列: {missing}")

    print(f"成功读取 {len(df)} 条预处理后的评论")

    # 2. 添加自定义词典
    jieba.add_word("冰雪大世界")
    jieba.add_word("哈尔滨")
    jieba.add_word("冰雪节")

    # 3. 加载停用词表
    stopwords = load_stopwords(STOPWORDS_PATH)
    print(f"停用词数量: {len(stopwords)}")

    # 4. 分词：对 content_clean 分词，得到 tokens 列
    print("开始分词...")
    df["tokens"] = df["content_clean"].astype(str).apply(jieba.lcut)

    # 5. 去停用词和无效词，得到 clean_tokens
    print("过滤停用词和无效词...")
    df["clean_tokens"] = df["tokens"].apply(lambda toks: filter_tokens(toks, stopwords))

    # 6. 词性标注：对 content_clean 做词性标注
    print("词性标注...")
    df["pos_tags"] = df["content_clean"].astype(str).apply(pos_tag)

    # 7. 从词性标注中提取名词
    print("抽取名词...")
    df["nouns"] = df["pos_tags"].apply(extract_nouns)

    # 8. 导出到新的 Excel
    df.to_excel(OUTPUT_EXCEL, index=False, sheet_name=OUTPUT_SHEET)
    print("已将分词/词性标注/名词抽取结果导出到:")
    print(OUTPUT_EXCEL)


if __name__ == "__main__":
    main()
