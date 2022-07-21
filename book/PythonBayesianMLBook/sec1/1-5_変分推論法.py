# ******************************************************************************
# Course  : Python出始めるベイズ機械学習入門
# Chapter : 1 ベイジアンモデリングとは
# Theme   : 5 変分推論法
# Date    : 2022/07/22
# Page    : P50 - P53
# URL     : https://github.com/sammy-suyama/PythonBayesianMLBook
# ******************************************************************************


# ＜概要＞
# - 変分推論法はサンプリングではなく最適化により近似を行うアプローチ
#   --- MCMCによる近似推論は広く用いられているが、計算コストの面で課題がある
#   --- 必ず事後分布を正確に表現できるという保証はない


# ＜目次＞
# 0 準備
# 1 シミュレーションデータの作成
# 2 シミュレーションの準備
# 3 シミュレーションの実行
# 4 収束モニタリング
# 5 事前分布と事後分布の比較


# 0 準備 ---------------------------------------------------------------------

# ライブラリ
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import stats

# インスタンス生成
tfd = tfp.distributions
tfb = tfp.bijectors

# プロット設定
sns.reset_defaults()
sns.set_context(context='talk', font_scale=0.8)
cmap = plt.get_cmap("tab10")


# 1 シミュレーションデータの作成 --------------------------------------------------

# ＜ポイント＞
# - ガウス分布から正規乱数を生成する
#   --- 変分推論法でガウス分布の平均パラメータの事後分布を求める


# パラメータの設定
# --- サンプリング数
# --- 平均の真値
# --- 標準偏差の真値
n_sample = 100
true_mean = 3.0
true_sd = 1.0

# シミュレーションデータの生成
# --- 正規乱数
np.random.seed(1)
data = np.random.normal(true_mean, true_sd, n_sample)

# データ確認
# --- 乱数の平均値と標準偏差
print('sample mean: {:.2f}, sample sd: {:.2f}'.format(data.mean(), data.std()))

# プロット作成
sns.displot(data, kde=False)
plt.xlabel('$x$')
plt.ylabel('count')
plt.show()


# 2 シミュレーションの準備 ----------------------------------------------------

# TFPによる変分推論
tf.random.set_seed(1)

# @title calculate posterior
lam = 1.0
lam_mu = 0.1
m = 0

#
lam_mu_hat = n_sample * lam + lam_mu
sigma_hat = np.sqrt(1 / lam_mu_hat)
m_hat = (lam * data.sum() + lam_mu * m) / lam_mu_hat

# インスタンス生成
Root = tfd.JointDistributionCoroutine.Root


# モデル定義
def model():
    mu = yield Root(tfd.Normal(loc=0, scale=10))  # 事前分布
    y = yield tfd.Sample(
        tfd.Normal(loc=mu, scale=true_sd),
        sample_shape=n_sample)


joint = tfd.JointDistributionCoroutine(model)


# yは観測ずみなので予め与えておく。muの正規化されていない事後分布になる。
# iidなサンプルn_sample個の確率の和をとる
def unnormalized_log_posterior(mu):
    return joint.log_prob(mu, data)


# 3 最適化の実行 ----------------------------------------------------

# インスタンス生成
q_mu_loc = tf.Variable(0., name='q_mu_loc')
q_mu_scale = tfp.util.TransformedVariable(1., tfb.Softplus(), name='q_mu_scale')
q_mu = tfd.Normal(loc=q_mu_loc, scale=q_mu_scale, name='q_mu')

# 最適化
losses = tfp.vi.fit_surrogate_posterior(
    unnormalized_log_posterior,
    surrogate_posterior=q_mu,
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    num_steps=200,
    sample_size=20,
    seed=1)


# 4 収束モニタリング ---------------------------------------------------------------------

# ＜ポイント＞
# - エビデンス下界(ELBO)を最小化することで事後分布との近似推論を行う


# データ確認
losses

# プロット
# --- ELBOの収束状況
plt.plot(losses)
plt.xlabel('steps')
plt.ylabel('negative ELBO')
plt.show()


# 5 事前分布と事後分布の比較 ----------------------------------------------------------

# プロット作成
xx = np.linspace(2.6, 3.5)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(xx, stats.norm(m_hat, sigma_hat).pdf(xx))
ax.plot(xx, stats.norm(q_mu_loc.numpy(), q_mu_scale.numpy()).pdf(xx))
ax.legend(['true posterior', 'estimation by VI'],
          loc='center left',
          bbox_to_anchor=(1.0, 0.5))
ax.set_xlabel('$\mu$')
ax.set_ylabel('Density')
plt.show()

# 真値と推定値の比較
print('true posterior mean: {:.3f}, sd: {:.3f}'.format(m_hat, sigma_hat))
print('estimated posterior mean: {:.3f}, sd: {:.3f}'.format(q_mu_loc.numpy(), q_mu_scale.numpy()))
