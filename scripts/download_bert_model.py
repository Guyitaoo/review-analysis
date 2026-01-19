import os
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "bert-base-chinese"
# 将完整模型保存到项目内固定目录，便于后续离线加载
# 获取项目根目录（scripts的父目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "model", "bert-base-chinese-local")

def main():
    print("=" * 60)
    print(f"准备下载并保存模型: {MODEL_NAME}")
    print("本地保存路径:", LOCAL_MODEL_DIR)
    print("=" * 60)

    # 1. 从 HuggingFace 下载（如果本地已有缓存，会直接复用缓存）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    # 2. 保存到项目内固定目录
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    model.save_pretrained(LOCAL_MODEL_DIR)

    print("模型已保存到本地目录（可离线使用）:")
    print(LOCAL_MODEL_DIR)
    print("下次在实验代码中，直接 from_pretrained(LOCAL_MODEL_DIR) 即可。")


if __name__ == "__main__":
    main()
