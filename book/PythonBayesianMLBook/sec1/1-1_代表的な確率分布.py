# ******************************************************************************
# Course  : Python出始めるベイズ機械学習入門
# Chapter : 1 ベイジアンモデリングとは
# Theme   : 代表的な確率分布
# Date    : 2022/07/13
# Page    : P8 - P27
# URL     : https://github.com/sammy-suyama/PythonBayesianMLBook
# ******************************************************************************


# ＜概要＞
# - ベイズ統計モデリングでは確率分布の知識を前提とした分野である
#   --- ベイズ統計モデリングはシミュレーションベースなので確率分布を前提とはしない
#   --- ただし、確率分布ごとの使いどころを理解しておく必要がある


# ＜目次＞
# 0 準備
# 1 ベルヌーイ分布
# 2 カテゴリ分布
# 3 二項分布
# 4 ポアソン分布
# 5 一様分布
# 6 一次元ガウス分布(正規分布)
# 7 多次元ガウス分布(多変量正規分布)
# 8 ベータ分布
# 9 ディリクレ分布
# 10 ガンマ分布


# 0 準備 ---------------------------------------------------------------------

# ライブラリ
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# 1 ベルヌーイ分布 -------------------------------------------------------------

# ＜ポイント＞
# - コインの表裏などTrue(1)/False(0)で表すことができるものはベルヌーイ分布として表現される
#   --- パラメータはTrueとなる確率θ


# ベルヌーイ分布のインスタンス生成
d = stats.bernoulli(0.6)

# 乱数生成
# --- 確率分布からデータを生成
X = d.rvs(100)
X

# 乱数における1の割合
# --- 0.6に近づくことが期待される
sum(X) / len(X)

# 確率質量分布
# --- ベルヌーイ分布の実現値は0/1の2通り
print(d.pmf(0), d.pmf(1))

# プロット作成
plt.bar([0, 1], d.pmf([0, 1]))
plt.show()

# 平均/分散
# --- 確率分布における理論値
print(d.mean(), d.var())


# 2 カテゴリ分布 ----------------------------------------------------------------

# ＜ポイント＞
# - カテゴリ分布はベルヌーイ分布を3カテゴリ以上に拡張した分布といえる
#   --- 乱数はOne-Hotで表現される


# カテゴリ分布のインスタンス生成
# --- N=1の多項分布としてカテゴリ分布を定義
# --- リストには発生確率を指定
cat_dist = stats.multinomial(1, [0.1, 0.2, 0.3, 0.4])

# 要素数の取得
K = len(cat_dist.p)

# 乱数生成
# --- 確率分布からデータを生成
X_onehot = cat_dist.rvs(100)
X_onehot

# 乱数取得
# --- One-Hot表現の1の位置を取得
X = [np.argmax(x) for x in X_onehot]

# プロット作成
# --- ヒストグラムが発生確率に基づいていることが確認できる
plt.hist(X, bins=range(K + 1))
plt.show()

# プロット作成
# --- 理論分布
X_tmp = np.identity(K)[range(K)]
X_tmp
plt.bar(range(K), cat_dist.pmf(X_tmp))
plt.show()


# 3 二項分布 -----------------------------------------------------------------

# ＜ポイント＞
# - 二項分布はベルヌーイ分布の多試行した場合の分布である
#   --- True/Falseの試行をN回行った場合の期待値を多試行した場合の分布


# パラメータ設定
N = 8
theta = 0.2

# 二項分布のインスタンス生成
bin_dist = stats.binom(N, theta)

# 乱数生成
# --- 定義された二項分布からサンプルを100個生成
X = bin_dist.rvs(100)
X

# ヒストグラムを作成
plt.hist(X, range(N + 1))
plt.show()

# サンプル統計量の確認
# --- 平均
# --- 分散
print('average = ' + str(np.mean(X)))
print('variance = ' + str(np.std(X) ** 2))

# プロット作成
# --- 理論分布
ar = np.arange(0, N + 1)
plt.bar(ar, bin_dist.pmf(ar))
plt.show()

# プロット作成
# --- θを変更した場合の分布
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
thetas = [0.1, 0.5, 0.9]
for i, theta in enumerate(thetas):
    bin_dist = stats.binom(N, theta)
    axes[i].bar(ar, bin_dist.pmf(ar))
    axes[i].set_title('mu = ' + str(theta))
plt.show()


# 4 ポアソン分布 -----------------------------------------------------------

# ＜ポイント＞
# - ポアソン分布は自然数(x=0,1,2,…)に関数分布のことを指す


# インスタンス生成
poi_dist = stats.poisson(3.0)

# 乱数生成
# --- 定義されたポアソン分布からサンプルを100個生成
X = poi_dist.rvs(100)
X

# ヒストグラムを描く
plt.hist(X, range(30))
plt.show()

# サンプル統計量の確認
# --- 平均
# --- 分散
print('average = ' + str(np.mean(X)))
print('variance = ' + str(np.std(X)**2))


# 確率質量関数のプロット（適当に30で打ち切る）
ar = np.arange(0,30)
plt.bar(ar, poi_dist.pmf(ar))
plt.show()

# ポアソン分布のパラメータを変えてみて，確率質量関数の変化を見てみる
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
params = [0.1, 1.0, 10.0]
for i, param in enumerate(params):
  poi_dist = stats.poisson(param)
  axes[i].bar(np.arange(0,30), poi_dist.pmf(np.arange(0,30)))
  axes[i].set_title('mu = ' + str(param))
plt.show()


# 5 一様分布 ---------------------------------------------------------------

# ＜ポイント＞
# -


# インスタンス生成
uni_dist = stats.uniform(2, 5-2)

# 乱数生成
# --- 定義された一様分布からサンプルを100個生成
X = uni_dist.rvs(100)
X

# ヒストグラムの作成
plt.hist(X, bins=10)
plt.show()


# 6 一次元ガウス分布(正規分布) -----------------------------------------------

# インスタンス生成
normal_dist = stats.norm(0.0, 1.0)

# 乱数生成
# --- 定義された正規分布からサンプルを100個生成
X = normal_dist.rvs(1000)
X

# ヒストグラムの作成
plt.hist(X, bins=10)
plt.show()

# 理論分布の作成
ls = np.linspace(-3, 3, 100)
plt.plot(ls, normal_dist.pdf(ls))
plt.show()

# シミュレーション
# --- 標準偏差を変えてガウス分布をプロットしてみる
mu = 0
sigma_list = [0.2, 0.5, 1.0, 2.0]
for sigma in sigma_list:
  normal_dist = stats.norm(mu, sigma)
  plt.plot(ls, normal_dist.pdf(ls),
           label='mu = ' + str(mu) + ', sigma = ' + str(sigma))
plt.legend()
plt.show()


# 7 多次元ガウス分布(多変量正規分布) -----------------------------------------

# 非対角成分が0の場合 ------------------------------------

# パラメータ設定
mu = [0, 0]
Sigma = [[1.0, 0.0],
         [0.0, 1.0]]

# インスタンス生成
mvn_dist = stats.multivariate_normal(mu, Sigma)

# 乱数生成
# --- 定義された多変量正規分布からサンプルを100個生成
X = mvn_dist.rvs(1000)
X

# 散布図作成
plt.scatter(X[:,0], X[:,1])
plt.show()


# 非対角成分が0でない場合 ---------------------------------

# パラメータ設定
mu = [0, 0]
Sigma = [[1.0, 0.5],
         [0.5, 1.0]]

# インスタンス生成
mvn_dist = stats.multivariate_normal(mu, Sigma)

# 乱数生成
# --- 定義された多変量正規分布からサンプルを100個生成
X = mvn_dist.rvs(1000)
X

# 散布図作成
plt.scatter(X[:,0], X[:,1])
plt.show()

# 等高線の作成
x1, x2 = np.mgrid[-3:3:.01, -3:3:.01]
pos = np.dstack((x1, x2))
plt.contourf(x1, x2, mvn_dist.pdf(pos))
plt.show()


# 8 ベータ分布 ------------------------------------------------------------

# インスタンス生成
beta_dist = stats.beta(0.1, 0.1)

# 乱数生成
# --- 定義されたベータ分布からサンプルを100個生成
X = beta_dist.rvs(1000)
X

# ヒストグラムの作成
plt.hist(X, bins=10)
plt.show()

# 確率密度関数のプロット
# --- 理論分布
ls = np.linspace(0, 1, 100)
plt.plot(ls, beta_dist.pdf(ls))
plt.show()

# ベータ分布の定義域は(0, 1)
ls = np.linspace(0,1,100)
plt.plot(ls, beta_dist.pdf(ls))
plt.show()

# シミュレーション
# --- 各パラメータを変えてベータ分布をプロットしてみる
alpha_list = [0.1, 1.0, 2.0]
beta_list = [0.1, 1.0, 2.0]
for alpha in alpha_list:
  for beta in beta_list:
    beta_dist = stats.beta(alpha, beta)
    plt.plot(ls, beta_dist.pdf(ls),
             label='alpha = ' + str(alpha) + ', beta = ' + str(beta))
plt.legend()
plt.show()


# 9 ディリクレ分布 ---------------------------------------------------------

# インスタンス生成
dir_dist = stats.dirichlet([0.5, 0.5, 0.5])

# 乱数生成
# --- 定義されたディリクレ分布からサンプルを100個生成
X = dir_dist.rvs(1000)
X

# 散布図を描く
plt.figure(figsize=(6, 6))
plt.scatter(X[:,0], X[:,1], alpha=0.1)
plt.show()

# シミュレーション
# --- αを変えてディリクレ分布をプロットしてみる
alpha_list = [[0.1, 0.1, 0.1], [1.0, 1.0, 1.0],
              [5.0, 5.0, 5.0], [0.1, 1.0, 5.0]]

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for (i, ax) in enumerate(axes.ravel()):
    alpha = alpha_list[i]
    dir_dist = stats.dirichlet(alpha)
    X = dir_dist.rvs(1000)
    ax.scatter(X[:,0], X[:,1], alpha=0.1)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_title('alpha = ' + str(alpha))
plt.show()


# 10 ガンマ分布 -----------------------------------------------------------

# パラメータ設定
alpha = 1.0
theta = 1.0

# インスタンス生成
gamma_dist = stats.gamma(a=alpha, scale=theta)

# 乱数生成
# --- 定義されたガンマ分布からサンプルを100個生成
X = gamma_dist.rvs(1000)
X

# ヒストグラム作成
plt.hist(X, bins=10)
plt.show()

# 確率密度関数
# --- 理論分布
ls = np.linspace(0, 3, 100)
plt.plot(ls, gamma_dist.pdf(ls))
plt.show()

# ガンマ分布の定義域（5で打ち切り）
ls = np.linspace(0,5,100)

# シミュレーション
# --- パラメータを変えてベータ分布をプロット
alpha_list = [0.5, 1.0, 2.0]
theta_list = [0.5, 1.0, 2.0]
for alpha in alpha_list:
  for theta in theta_list:
    gamma_dist = stats.gamma(a=alpha, scale=theta)
    plt.plot(ls, gamma_dist.pdf(ls),
             label='alpha = ' + str(alpha) + ', theta = ' + str(theta))
plt.legend()
plt.show()
