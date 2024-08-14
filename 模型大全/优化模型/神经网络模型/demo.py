import tensorflow as tf
import numpy as np


class PointerNetwork(tf.keras.Model):
    def __init__(self, hidden_size):
        super(PointerNetwork, self).__init__()
        self.encoder = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True)
        self.decoder = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True)
        self.attention = tf.keras.layers.Attention()
        self.pointer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        encoder_outputs, state_h, state_c = self.encoder(inputs)
        decoder_outputs, _, _ = self.decoder(encoder_outputs, initial_state=[state_h, state_c])
        attention_outputs = self.attention([decoder_outputs, encoder_outputs])
        pointer_logits = self.pointer(attention_outputs)
        return pointer_logits


# 生成随机TSP问题
def generate_tsp_data(num_cities, num_samples):
    data = np.random.rand(num_samples, num_cities, 2)  # 2D coordinates
    return data


# 准备数据
num_cities = 10
num_samples = 1000
tsp_data = generate_tsp_data(num_cities, num_samples)

# 创建和编译模型
model = PointerNetwork(hidden_size=128)
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(tsp_data, tsp_data, epochs=100, batch_size=32)


# 使用模型生成路径
def generate_path(model, cities):
    cities_tensor = tf.convert_to_tensor(cities, dtype=tf.float32)
    cities_tensor = tf.expand_dims(cities_tensor, 0)

    path = []
    for _ in range(len(cities)):
        logits = model(cities_tensor)
        probabilities = tf.nn.softmax(tf.squeeze(logits, -1))
        next_city = tf.argmax(probabilities, axis=-1)
        path.append(next_city.numpy()[0])

        mask = tf.one_hot(next_city, depth=tf.shape(cities_tensor)[1])
        cities_tensor = cities_tensor * (1 - mask) - 1e9 * mask

    return path


# 测试模型
test_cities = np.random.rand(num_cities, 2)
generated_path = generate_path(model, test_cities)
print("Generated path:", generated_path)
