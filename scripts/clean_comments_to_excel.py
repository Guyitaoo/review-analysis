"""清洗景区评论并导出为 Excel。

步骤：
1. 从原始 Excel 中读取评论数据（sheet: scenic_comments）。
2. 清理表情符号、URL、@、话题、特殊字符、数字、多余空格等，生成 content_clean 列。
3. 按「username + content_clean」去重，保留第一条。
4. 将去重后的结果导出到 result/preprocessing/ 文件夹。
"""

import os
import re
import emoji
import pandas as pd


# 原始数据文件路径
INPUT_EXCEL = os.path.join("assets", "scenic_comments_harbin_snowworld_ly.xlsx")
INPUT_SHEET = "scenic_comments"

# 输出文件夹
OUTPUT_DIR = os.path.join("result", "preprocessing")

# 清洗后的结果导出路径
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "scenic_comments_cleaned_dedup.xlsx")
OUTPUT_SHEET = "scenic_comments_cleaned"


def remove_emoji_and_special(text: str) -> str:
    """清理表情符号和特殊字符，对应原 read_file.py 中 remove_emoji 逻辑。"""
    if pd.isna(text):
        return ""
    text = str(text)
    # 去除 emoji
    text = emoji.replace_emoji(text, replace="")
    # 去除特殊符号、URL、@、话题等
    text = re.sub(r"[①-⑳]", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"#\S+", "", text)
    # 只保留中文、英文、数字和空白
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", "", text)
    # 去除数字
    text = re.sub(r"\d+", "", text)
    # 规整空白字符
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_duplicate_comments(df: pd.DataFrame) -> pd.DataFrame:
    """按用户名 + 清洗后的评论内容去重，保留第一条。"""
    dup_mask = df.duplicated(subset=["username", "content_clean"], keep="first")
    dup_count = int(dup_mask.sum())

    print(f"重复评论条数：{dup_count}")

    dedup_df = (
        df.drop_duplicates(subset=["username", "content_clean"], keep="first")
        .reset_index(drop=True)
    )

    print(f"去重前共有 {len(df)} 条评论，去重后剩余 {len(dedup_df)} 条。")
    return dedup_df


def main():
    print("=" * 60)
    print("读取原始评论数据并进行清洗 + 去重")
    print("输入文件:", INPUT_EXCEL)
    print("输出文件:", OUTPUT_EXCEL)
    print("=" * 60)
    
    # 创建输出文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出文件夹: {OUTPUT_DIR}")

    # 1. 读取数据，只保留需要的三列
    df = pd.read_excel(INPUT_EXCEL, sheet_name=INPUT_SHEET)
    df = df[["username", "rating", "content"]]
    print(f"成功读取 {len(df)} 条原始评论")

    # 2. 清洗文本，生成 content_clean 列
    df["content_clean"] = df["content"].apply(remove_emoji_and_special)

    # 3. 去重
    df_dedup = remove_duplicate_comments(df)

    # 4. 删除原始 content 列，只保留清洗后的 content_clean
    df_dedup = df_dedup.drop(columns=["content"], errors="ignore")
    # 可选：将 content_clean 重命名为 content（如果希望列名更简洁）
    # df_dedup = df_dedup.rename(columns={"content_clean": "content"})

    # 5. 导出到新的 Excel
    df_dedup.to_excel(OUTPUT_EXCEL, index=False, sheet_name=OUTPUT_SHEET)
    print("已将清洗+去重后的数据导出到:")
    print(OUTPUT_EXCEL)


if __name__ == "__main__":
    main()
