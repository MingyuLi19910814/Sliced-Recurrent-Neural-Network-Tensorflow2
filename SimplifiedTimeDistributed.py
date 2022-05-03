import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import tensor_shape

class ModelTimeDistributed(tf.keras.layers.Wrapper):
    '''
    Simplified module of tf.keras.layers.TimeDistributed for tf.keras.Model,
    since in current Tensorflow version, the TimeDistributed Model has bug when computing output shape of tf.keras.Model or GRU module
    '''
    def __init__(self, layer, **kwargs):
        if not isinstance(layer, tf.keras.Model):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`tf.keras.layers.Layer` instance. You passed: {input}'.format(
                    input=layer))
        super(ModelTimeDistributed, self).__init__(layer, **kwargs)

    def _get_shape_tuple(self, init_tuple, tensor, start_idx, int_shape=None):
        # replace all None in int_shape by K.shape
        if int_shape is None:
            int_shape = K.int_shape(tensor)[start_idx:]
        if not any(not s for s in int_shape):
            return init_tuple + tuple(int_shape)
        shape = K.shape(tensor)
        int_shape = list(int_shape)
        for i, s in enumerate(int_shape):
            if not s:
                int_shape[i] = shape[start_idx + i]
        return init_tuple + tuple(int_shape)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        child_input_shape = tensor_shape.TensorShape([input_shape[0]] +
                                                     input_shape[2:])
        # child_output_shape = self.layer.compute_output_shape(child_input_shape)
        # if not isinstance(child_output_shape, tensor_shape.TensorShape):
        #     child_output_shape = tensor_shape.TensorShape(child_output_shape)
        # child_output_shape = child_output_shape.as_list()
        timesteps = input_shape[1]
        return tensor_shape.TensorShape([child_input_shape[0], timesteps] +
                                        list(self.layer.output_shape[1:]))

    def call(self, inputs, training=None, mask=None):
        kwargs = {}
        if generic_utils.has_arg(self.layer.call, 'training'):
            kwargs['training'] = training

        input_shape = K.int_shape(inputs)
        input_length = input_shape[1]
        if not input_length:
            input_length = array_ops.shape(inputs)[1]
        inner_input_shape = self._get_shape_tuple((-1,), inputs, 2)
        # Shape: (num_samples * timesteps, ...). And track the
        # transformation in self._input_map.
        inputs = array_ops.reshape(inputs, inner_input_shape)
        # (num_samples * timesteps, ...)
        if generic_utils.has_arg(self.layer.call, 'mask') and mask is not None:
            inner_mask_shape = self._get_shape_tuple((-1,), mask, 2)
            kwargs['mask'] = K.reshape(mask, inner_mask_shape)

        y = self.layer(inputs, **kwargs)

        # Shape: (num_samples, timesteps, ...)
        output_shape = self.compute_output_shape(input_shape).as_list()
        output_shape = self._get_shape_tuple((-1, input_length), y, 1,
                                             output_shape[2:])
        y = array_ops.reshape(y, output_shape)

        return y