import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = [1, 2, 3, 4, 5]
ys = [2, 4, 6, 8, 10]

model.fit(xs, ys, epochs=1000)

print(model.predict([4333]))
