import torch
import subprocess

print("="*50)
print("GPU诊断信息")
print("="*50)
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# 检查PyTorch是否支持CUDA
if torch.cuda.is_available():
    print(f"✅ PyTorch支持CUDA")
    if torch.version.cuda:
        print(f"PyTorch编译的CUDA版本: {torch.version.cuda}")
    if torch.cuda.device_count() > 0:
        print(f"检测到 {torch.cuda.device_count()} 个GPU设备")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    # 进一步检查是否安装了CUDA版本的PyTorch
    if hasattr(torch.version, 'cuda') and torch.version.cuda:
        print(f"⚠ PyTorch编译时支持CUDA版本: {torch.version.cuda}")
        print("⚠ 但当前环境无法使用CUDA（可能是驱动问题或CUDA未正确安装）")
    else:
        print("⚠ PyTorch是CPU版本，没有CUDA支持")

print("\n系统GPU信息:")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    print(result.stdout)
except:
    print("无法运行nvidia-smi")
