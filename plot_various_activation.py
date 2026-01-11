import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置风格
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

def sigmoid(x, k=1):
    return 1 / (1 + np.exp(-k * x))

def bounded_relu(x, tau, k=1.5):
    # ReLU(Tanh((x - tau) * k))
    val = np.tanh((x - tau) * k)
    return np.maximum(0, val)

def standard_relu(x, tau):
    return np.maximum(0, x - tau)

# 数据准备
x = np.linspace(-2, 4, 500)
tau = 1.0  # 假设阈值是 1.0

# 计算曲线
y_sigmoid = sigmoid(x - tau, k=4) # Shifted Sigmoid
y_hard = np.where(x > tau, 1.0, 0.0) # Original SPARC Hard Gate
y_relu = standard_relu(x, tau) # Unbounded ReLU
y_bounded = bounded_relu(x, tau, k=2) # Ours

# 绘图
plt.figure(figsize=(10, 6))

# 1. 绘制区域背景 (Noise vs Signal)
plt.axvspan(-2, tau, color='grey', alpha=0.1, label='Background / Noise Region')
plt.axvspan(tau, 4, color='green', alpha=0.05, label='Salient / Signal Region')

# 2. 绘制曲线
# Sigmoid - 展示 Leakage
plt.plot(x, y_sigmoid, label='Sigmoid (Soft)', linestyle='--', color='orange', linewidth=2, alpha=0.7)

# Standard ReLU - 展示 Unbounded
plt.plot(x, y_relu, label='Standard ReLU (Unbounded)', linestyle=':', color='red', linewidth=2.5)

# Hard Threshold - Baseline
plt.plot(x, y_hard, label='Hard Threshold (Baseline)', linestyle='-.', color='black', linewidth=1.5, alpha=0.6)

# Ours
plt.plot(x, y_bounded, label='Bounded ReLU (Ours)', color='blue', linewidth=3)

# 3. 添加标注 (Annotations)
# 标注 Sigmoid 的 Leakage
plt.annotate('Noise Leakage\n(Sparsity Loss)', xy=(0, 0.05), xytext=(-1.5, 0.3),
             arrowprops=dict(facecolor='orange', shrink=0.05), color='darkorange', fontweight='bold')

# 标注 ReLU 的 Explosion
plt.annotate('Signal Explosion\n(Repetition Risk)', xy=(2.5, 1.5), xytext=(1.5, 2.0),
             arrowprops=dict(facecolor='red', shrink=0.05), color='red', fontweight='bold')

# 标注 Ours 的 Saturation
plt.annotate('Safe Saturation', xy=(3, 1.0), xytext=(3.2, 0.8),
             arrowprops=dict(facecolor='blue', shrink=0.05), color='blue', fontweight='bold')

# 装饰
plt.axvline(tau, color='black', linestyle='--', alpha=0.3)
plt.text(tau+0.05, -0.1, r'Threshold $\tau$', fontsize=12)

plt.title('Comparison of Activation Functions for Visual Attention Calibration', fontsize=14, pad=20)
plt.xlabel(r'Relative Attention Ratio ($r_i$)', fontsize=12)
plt.ylabel('Activation Output', fontsize=12)
plt.ylim(-0.1, 2.2)
plt.xlim(-2, 4)
plt.legend(loc='upper left', frameon=True)

# 保存
plt.tight_layout()
plt.savefig('activation_comparison.png', dpi=300)
plt.show()