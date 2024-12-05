import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models

data = pd.read_csv("data/BTCUSDT_2024.7.18to2024.8.3.csv")
kline_data = data[['Open', 'High', 'Low', 'Close']].values  # 转换为 numpy 数组


# 设置随机噪声和标签的维度
latent_dim = 100  # 噪声维度
num_classes = 3   # 假设有 3 种结构类型（例如：头肩顶、双底等）

# 构建生成器
def build_generator():
    noise_input = layers.Input(shape=(latent_dim,))
    label_input = layers.Input(shape=(num_classes,))  # 条件输入（形态标签）

    # 将噪声和标签拼接
    combined_input = layers.Concatenate()([noise_input, label_input])

    # 生成器网络结构
    x = layers.Dense(128, activation="relu")(combined_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(4, activation="linear")  # 输出 K 线数据（Open, High, Low, Close）

    return models.Model([noise_input, label_input], x)

# 构建判别器
def build_discriminator():
    data_input = layers.Input(shape=(4,))  # 输入 K 线数据（Open, High, Low, Close）
    label_input = layers.Input(shape=(num_classes,))  # 条件输入

    # 将数据和标签拼接
    combined_input = layers.Concatenate()([data_input, label_input])

    # 判别器网络结构
    x = layers.Dense(256, activation="relu")(combined_input)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1, activation="sigmoid")  # 输出真假概率

    model = models.Model([data_input, label_input], x)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# 创建生成器和判别器
generator = build_generator()
discriminator = build_discriminator()


# 创建 cGAN 模型
discriminator.trainable = False  # 在训练生成器时，判别器不更新
noise_input = layers.Input(shape=(latent_dim,))
label_input = layers.Input(shape=(num_classes,))
generated_data = generator([noise_input, label_input])
validity = discriminator([generated_data, label_input])

cgan = models.Model([noise_input, label_input], validity)
cgan.compile(optimizer="adam", loss="binary_crossentropy")


# 标签
real = np.ones((batch_size, 1))  # 真标签
fake = np.zeros((batch_size, 1))  # 假标签

# 训练过程
def train_cgan(generator, discriminator, cgan, epochs, batch_size, kline_data, labels):
    for epoch in range(epochs):
        # 随机选择一个批次的真实数据
        idx = np.random.randint(0, kline_data.shape[0], batch_size)
        real_data = kline_data[idx]
        real_labels = labels[idx]

        # 生成假数据
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_labels = real_labels  # 使用真实标签作为条件
        fake_data = generator.predict([noise, fake_labels])

        # 训练判别器
        d_loss_real = discriminator.train_on_batch([real_data, real_labels], real)
        d_loss_fake = discriminator.train_on_batch([fake_data, fake_labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # 训练生成器
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = cgan.train_on_batch([noise, fake_labels], real)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")

# 训练
epochs = 10000
batch_size = 64
train_cgan(generator, discriminator, cgan, epochs, batch_size, kline_data, labels)



# 生成特定形态的 K 线数据
noise = np.random.normal(0, 1, (10, latent_dim))  # 生成 10 个噪声向量
target_label = np.zeros((10, num_classes))  # 假设目标是第 0 类（头肩顶）
target_label[:, 0] = 1

generated_kline = generator.predict([noise, target_label])
print("生成的 K 线数据：")
print(generated_kline)
