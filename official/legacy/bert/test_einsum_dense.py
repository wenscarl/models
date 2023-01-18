import tensorflow as tf
from keras.layers import core
import EinsumDenseFp8 as einsum_dense_fp8

# all dims are multiple of 16
B, C, D, E = 16, 32, 48, 64

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Enable XLA.
tf.keras.layers.EinsumDense = einsum_dense_fp8.EinsumDenseFp8

# Please tweak use_variable, if set to False, you will see fp8 gemm, otherwise not. It's not expected.
class TestModel(tf.keras.Model):
    def build(self, ouptut_shape):
        self.einsumdense=tf.keras.layers.EinsumDense(
            'abc,cde->abde',output_shape=(B, D, E), use_variable=True)
    def call (self, inputs):
        x =self.einsumdense(inputs)
        return x
model = TestModel()

bs = 96
x_data = tf.random.normal(shape=(bs, B, C))
y_data = tf.random.normal(shape=(bs, B, D, E))

model.compile(
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=["accuracy"],
#    run_eagerly=True,
    jit_compile=True,
)
history = model.fit(x_data, y_data, batch_size=32, epochs=200,
                    validation_split=0.2, verbose=1)
