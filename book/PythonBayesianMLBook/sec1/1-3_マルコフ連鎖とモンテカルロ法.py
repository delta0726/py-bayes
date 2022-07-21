# ******************************************************************************
# Course  : Python出始めるベイズ機械学習入門
# Chapter : 1 ベイジアンモデリングとは
# Theme   : 3 マルコフ連鎖とモンテカルロ法
# Date    : 2022/07/21
# Page    : P33 - P37
# URL     : https://github.com/sammy-suyama/PythonBayesianMLBook
# ******************************************************************************


# ＜概要＞
# - マルコフ連鎖モンテカルロ法法を理解する準備として｢マルコフ連鎖｣と｢モンテカルロ法｣を確認する


# ＜目次＞
# 0 準備
# 1 マルコフ連鎖
# 2 モンテカルロ法


# 0 準備 ---------------------------------------------------------------------

# ライブラリ
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import tensorflow_probability as tfp
from scipy import stats

# インスタンス生成
tfd = tfp.distributions
tfb = tfp.bijectors


# 1 マルコフ連鎖 --------------------------------------------------------------------

# ＜ポイント＞
# - マルコフ連鎖とは、1つ前の状態から次の状態が決まることをいう
#   --- 遷移確率が定義され遷移行列として表される
#   --- 十分な期間を経ると定常分布に基づき定常状態となる（エルゴード性）


# 遷移確率
T = np.array([[0.2, 0.3, 0.5], [0.1, 0.5, 0.4], [0.1, 0.3, 0.6]])

# 初期状態
init_1 = np.array([0.6, 0.3, 0.1])
init_2 = np.array([0, 0.9, 0.1])


# 関数定義
# --- マルコフレ連鎖の推移プロット(P35)
def plot_markov_chain(init, T, n_steps):
    res = [init]
    for j in range(n_steps):
        init = init @ T
        res.append(init)
    res = np.array(res)

    plt.plot(res[:, 0], marker='o', label='A')
    plt.plot(res[:, 1], marker='o', label='B')
    plt.plot(res[:, 2], marker='o', label='C')
    plt.ylim(0, 1)
    # plt.legend()


# プロット1
plt.figure(figsize=(12, 4))
plt.subplot(121)
plot_markov_chain(init=init_1, T=T, n_steps=12)
plt.xlabel('Month')
plt.ylabel('Probability')
plt.title('initial distribution: $\pi^{(0)}_1=$' + f'{init_1}')

# プロット2
plt.subplot(122)
plot_markov_chain(init=init_2, T=T, n_steps=12)
plt.xlabel('Month')
plt.ylabel('Probability')
plt.title('initial distribution: $\pi^{(0)}_2=$' + f'{init_2}')

# プロット表示
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()


# 2 モンテカルロ法 --------------------------------------------------------------------

# ＜ポイント＞
# - モンテカルロ法とは、乱数を用いた数値計算法の総称のことをいう


# プロット
# --- 正規分布の確率密度関数
xx = np.linspace(-3, 3)
plt.plot(xx, stats.norm(0, 1).pdf(xx))

# プロット
# --- 分布範囲の色付け
xx = np.linspace(-2, 2)
plt.fill_between(xx, stats.norm(0, 1).pdf(xx), alpha=0.5)

# プロット表示
plt.xlabel("$z$")
plt.ylabel("$f(z)$")
plt.show()


# 関数定義
# --- モンテカルロ法
def montecarlo(M, trial=100, seed=1):
    """
    - M個のサンプルによるモンテカルロ法をtrial回繰り返し、推定値の平均と標準偏差を返す
    - 初期分布を一様分布(無情報事前分布)に設定する
    """
    np.random.seed(seed)
    x = np.random.uniform(-2, 2, (M, trial))
    res = 4 * stats.norm(0, 1).pdf(x).mean(axis=0)
    return res.mean(), res.std()


# 求めたい積分値の真値
# --- 検証用
ground_truth = stats.norm(0, 1).cdf(2) - stats.norm(0, 1).cdf(-2)
print(f'ground truth: {ground_truth:.4f}')

# シミュレーション
# --- 回数を増やすほど真値に収斂している
M_list = [100, 1000, 10000, 100000]
fig, ax = plt.subplots()
for M in M_list:
    m, s = montecarlo(M)
    print(f'M = {M}, estimation: {m:.4f} ± {s:.4f}')
    ax.scatter(M, m, c='b', marker='x')
    ax.errorbar(M, m, s, capsize=3, c="b")
ax.axhline(ground_truth, c="r", label="ground truth")
ax.set_xscale("log")
ax.legend()
ax.set_xlabel("sample size")
ax.set_ylabel("estimation")
plt.show()
