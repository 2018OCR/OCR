from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Activation


# the activation function
def squash(x, axis=-1):
    x_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    coefficient = K.sqrt(x_squared_norm) / (1.0 + x_squared_norm)
    return coefficient * x


# the routing_softmax function
def routing_softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


# Capsule Layer
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]

        self.W = self.add_weight(name='capsule_kernel',
                                 shape=(1, input_dim_capsule,
                                        self.num_capsule * self.dim_capsule),
                                 initializer='glorot_uniform',
                                 trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])
        for i in range(self.routings):
            c = routing_softmax(b, 1)
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return None, self.num_capsule, self.dim_capsule
