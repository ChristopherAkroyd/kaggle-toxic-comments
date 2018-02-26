from keras import backend as K
from keras.engine.topology import Layer
from keras import constraints, initializers, regularizers
from .attention_utils import dot_product


# http://colinraffel.com/publications/iclr2016feed.pdf
class FeedForwardAttention(Layer):
    def __init__(self, kernel_regularizer=None, bias_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 use_bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.use_bias = use_bias
        super(FeedForwardAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.kernel = self.add_weight((input_shape[-1],),
                                      initializer=self.init,
                                      name='{}_W'.format(self.name),
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight((input_shape[1],),
                                        initializer='zero',
                                        name='{}_b'.format(self.name),
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.trainable_weights = [self.kernel]
        self.built = True

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.kernel)

        if self.use_bias:
            eij += self.bias

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask.
        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        # Calculate the weighted sum.
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
