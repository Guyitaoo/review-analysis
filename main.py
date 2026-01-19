"""主程序：按照README中的步骤依次执行所有处理流程。

执行顺序：
1. 开发环境验证（可选）
2. 预加载：下载BERT模型
3. 预处理：数据清洗、分词、BERT向量化、词频统计
4. 聚类分析：基于词频和文档频的聚类
"""

import os
import sys
import subprocess
from pathlib import Path

# 添加scripts文件夹到Python路径
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# 脚本执行顺序（按照README.md中的顺序）
SCRIPTS = [
    # 开发环境验证（可选，默认跳过）
    # ("check_gpu.py", "检测GPU是否可用", False),
    
    # 预加载
    ("download_bert_model.py", "下载BERT模型到本地", True),
    
    # 预处理
    ("clean_comments_to_excel.py", "清洗和去重评论数据", True),
    ("tokenize_and_pos_tag_comments.py", "分词、去停用词、词性标注", True),
    ("bert_embed_words_with_pos.py", "生成BERT词向量", True),
    ("word_frequency_doc_freq_by_pos.py", "计算词频和文档频", True),
    
    # 聚类分析
    ("cluster_by_freq_top1000_nouns.py", "基于词频的聚类分析（Top1000名词）", True),
    ("cluster_by_doc_freq_top500.py", "基于文档频的聚类分析（Top500名词）", True),
]


def run_script(script_name, description, required=True):
    """
    运行指定的Python脚本
    
    参数:
        script_name: 脚本文件名
        description: 脚本描述
        required: 是否必需（如果失败是否继续）
    
    返回:
        bool: 是否成功执行
    """
    script_path = SCRIPTS_DIR / script_name
    
    if not script_path.exists():
        print(f"✗ 错误: 找不到脚本文件 {script_path}")
        return False
    
    print("\n" + "=" * 60)
    print(f"执行: {description}")
    print(f"脚本: {script_name}")
    print("=" * 60)
    
    try:
        # 切换到项目根目录执行脚本（保持工作目录在根目录）
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path(__file__).parent,
            check=False,
            capture_output=False
        )
        
        if result.returncode == 0:
            print(f"\n✓ {description} 完成")
            return True
        else:
            print(f"\n✗ {description} 失败 (退出码: {result.returncode})")
            if required:
                print(f"  这是必需步骤，程序将停止")
            else:
                print(f"  这是可选步骤，继续执行下一个")
            return False
            
    except Exception as e:
        print(f"\n✗ 执行 {script_name} 时出错: {str(e)}")
        if required:
            print(f"  这是必需步骤，程序将停止")
        return False


def main():
    """主函数：按顺序执行所有脚本"""
    print("=" * 60)
    print("旅游评论分析 - 主程序")
    print("=" * 60)
    print(f"\n项目根目录: {Path(__file__).parent}")
    print(f"脚本目录: {SCRIPTS_DIR}")
    print(f"\n将按顺序执行 {len(SCRIPTS)} 个步骤")
    
    # 确认是否继续
    print("\n" + "-" * 60)
    response = input("是否开始执行？(y/n，默认y): ").strip().lower()
    if response and response != 'y':
        print("已取消执行")
        return
    
    # 按顺序执行所有脚本
    failed_scripts = []
    
    for i, (script_name, description, required) in enumerate(SCRIPTS, 1):
        print(f"\n[{i}/{len(SCRIPTS)}] ", end="")
        success = run_script(script_name, description, required)
        
        if not success:
            if required:
                print("\n" + "=" * 60)
                print("程序执行中断")
                print("=" * 60)
                print(f"\n失败的步骤: {description} ({script_name})")
                if failed_scripts:
                    print(f"\n之前失败的步骤:")
                    for failed in failed_scripts:
                        print(f"  - {failed}")
                return
            else:
                failed_scripts.append(f"{description} ({script_name})")
    
    # 所有步骤完成
    print("\n" + "=" * 60)
    print("所有步骤执行完成！")
    print("=" * 60)
    
    if failed_scripts:
        print("\n以下可选步骤执行失败（不影响主流程）:")
        for failed in failed_scripts:
            print(f"  - {failed}")
    
    print("\n生成的文件位置:")
    print("  - 预处理结果: result/preprocessing/")
    print("  - 聚类结果: result/clustering/")
    print("  - BERT向量: word_embeddings_bert_with_pos.pkl")


if __name__ == "__main__":
    main()
