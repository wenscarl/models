# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras-based einsum dense layer."""


import re

import tensorflow.compat.v2 as tf

from keras import activations
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine.base_layer import Layer

import keras.backend as K
# isort: off
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.framework import dtypes

init_val_scalar = 0.8

FAKE_E4M3 = dtypes.float8_e4m3fn
FAKE_E5M2 = dtypes.float8_e5m2

E4M3_MAX = 448.
E5M2_MAX = 57344.
AMAX_HIS_LEN = 16


def get_fp8_max(fake_dtype):
  if fake_dtype == FAKE_E4M3:
    return E4M3_MAX
  else:
    assert fake_dtype == FAKE_E5M2
    return E5M2_MAX


def quantize(x, quantized_dtype, scale):
  dtype_max = get_fp8_max(quantized_dtype)
  scaled_x = tf.clip_by_value(x / scale, -dtype_max, dtype_max)
  return tf.cast(scaled_x, quantized_dtype)


def dequantize(x, wide_dtype, scale):
  return tf.cast(x, wide_dtype) * scale


def quantize_dequantize(x, quantized_dtype, scale):
  orig_dtype = x.dtype
  qx = quantize(x, quantized_dtype, scale)
  return dequantize(qx, orig_dtype, scale)

def update_scale(x, quantized_dtype, scale_var, amax_history):
  dtype_max = get_fp8_max(quantized_dtype)
  amax_current = tf.cast(tf.math.reduce_max(tf.math.abs(x)), scale_var.dtype)
  amax_his_tsr = tf.tensor_scatter_nd_update(tf.roll(amax_history.read_value(), 1, 0),[[0]],[amax_current])
  amax_history.assign(amax_his_tsr)
  amax_temp = tf.reduce_max(amax_history, axis=0)
  amax = tf.maximum(amax_temp, 2 ** -10)
  scale_var.assign(1.1 * amax / dtype_max)

def qdq_and_update(x, dtype, scale_var, amax_history):
  qx = quantize_dequantize(x, dtype, scale_var)
  update_scale(x, dtype, scale_var, amax_history)
  return qx

@keras_export(
    "keras.layers.EinsumDense", "keras.layers.experimental.EinsumDense"
)
class EinsumDenseFp8(Layer):
    """A layer that uses `tf.einsum` as the backing computation.

    This layer can perform einsum calculations of arbitrary dimensionality.

    Args:
      equation: An equation describing the einsum to perform. This equation must
        be a valid einsum string of the form `ab,bc->ac`, `...ab,bc->...ac`, or
        `ab...,bc->ac...` where 'ab', 'bc', and 'ac' can be any valid einsum
        axis expression sequence.
      output_shape: The expected shape of the output tensor (excluding the batch
        dimension and any dimensions represented by ellipses). You can specify
        None for any dimension that is unknown or can be inferred from the input
        shape.
      activation: Activation function to use. If you don't specify anything, no
        activation is applied (that is, a "linear" activation: `a(x) = x`).
      bias_axes: A string containing the output dimension(s) to apply a bias to.
        Each character in the `bias_axes` string should correspond to a
        character in the output portion of the `equation` string.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation").
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix.
      bias_constraint: Constraint function applied to the bias vector.

    Examples:

    **Biased dense layer with einsums**

    This example shows how to instantiate a standard Keras dense layer using
    einsum operations. This example is equivalent to
    `tf.keras.layers.Dense(64, use_bias=True)`.

    >>> layer = tf.keras.layers.EinsumDense("ab,bc->ac",
    ...                                     output_shape=64,
    ...                                     bias_axes="c")
    >>> input_tensor = tf.keras.Input(shape=[32])
    >>> output_tensor = layer(input_tensor)
    >>> output_tensor
    <... shape=(None, 64) dtype=...>

    **Applying a dense layer to a sequence**

    This example shows how to instantiate a layer that applies the same dense
    operation to every element in a sequence. Here, the `output_shape` has two
    values (since there are two non-batch dimensions in the output); the first
    dimension in the `output_shape` is `None`, because the sequence dimension
    `b` has an unknown shape.

    >>> layer = tf.keras.layers.EinsumDense("abc,cd->abd",
    ...                                     output_shape=(None, 64),
    ...                                     bias_axes="d")
    >>> input_tensor = tf.keras.Input(shape=[32, 128])
    >>> output_tensor = layer(input_tensor)
    >>> output_tensor
    <... shape=(None, 32, 64) dtype=...>

    **Applying a dense layer to a sequence using ellipses**

    This example shows how to instantiate a layer that applies the same dense
    operation to every element in a sequence, but uses the ellipsis notation
    instead of specifying the batch and sequence dimensions.

    Because we are using ellipsis notation and have specified only one axis, the
    `output_shape` arg is a single value. When instantiated in this way, the
    layer can handle any number of sequence dimensions - including the case
    where no sequence dimension exists.

    >>> layer = tf.keras.layers.EinsumDense("...x,xy->...y",
    ...                                     output_shape=64,
    ...                                     bias_axes="y")
    >>> input_tensor = tf.keras.Input(shape=[32, 128])
    >>> output_tensor = layer(input_tensor)
    >>> output_tensor
    <... shape=(None, 32, 64) dtype=...>
    """

    def __init__(
        self,
        equation,
        output_shape,
        activation=None,
        bias_axes=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_variable=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.equation = equation
        if isinstance(output_shape, int):
            self.partial_output_shape = [output_shape]
        else:
            self.partial_output_shape = list(output_shape)
        self.bias_axes = bias_axes
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.use_variable = use_variable

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        shape_data = _analyze_einsum_string(
            self.equation,
            self.bias_axes,
            input_shape,
            self.partial_output_shape,
        )
        kernel_shape, bias_shape, self.full_output_shape = shape_data
        self.kernel = self.add_weight(
            "kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )

        if bias_shape is not None:
            self.bias = self.add_weight(
                "bias",
                shape=bias_shape,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        init_val = tf.keras.initializers.Constant(init_val_scalar)
        self.input_amax_history = self.add_weight(
            "input_amax_history", shape=(AMAX_HIS_LEN,),
            initializer=init_val, trainable=False)
        self.input_scale = self.add_weight("input_scale", shape=(),
                                           initializer=init_val, trainable=False)
        self.kernel_amax_history = self.add_weight(
            "kernel_amax_history", shape=(AMAX_HIS_LEN,),
            initializer=init_val, trainable=False)
        self.kernel_scale = self.add_weight("kernel_scale", shape=(),
                                            initializer=init_val, trainable=False)
        self.input_grad_amax_history = self.add_weight(
            "input_grad_amax_history", shape=(AMAX_HIS_LEN,),
            initializer=init_val, trainable=False)
        self.input_grad_scale = self.add_weight("input_grad_scale", shape=(),
                                                initializer=init_val,
                                                trainable=False)
        self.output_grad_amax_history = self.add_weight(
            "output_grad_amax_history", shape=(AMAX_HIS_LEN,),
            initializer=init_val, trainable=False)
        self.output_grad_scale = self.add_weight(
            "output_grad_scale", shape=(),
            initializer=init_val, trainable=False)
        super().build(input_shape)

    def compute_output_shape(self, _):
        return tf.TensorShape(self.full_output_shape)

    def get_config(self):
        config = {
            "output_shape": self.partial_output_shape,
            "equation": self.equation,
            "activation": activations.serialize(self.activation),
            "bias_axes": self.bias_axes,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf.custom_gradient
    def in_qdq(self, input):
      """Quantize-dequantize both the input and the input's gradient."""
      qin = qdq_and_update(input, FAKE_E4M3, self.input_scale, self.input_amax_history)

      def grad(in_grad):
        in_grad_ret = qdq_and_update(in_grad, FAKE_E5M2, self.input_grad_scale, 
                                     self.input_grad_amax_history)
        return in_grad_ret
  
      return qin, grad
  
    @tf.custom_gradient
    def identity_qdq(self, output):
      """Quantize-dequantize both the output and the output's gradient, only if the next layer(in fwd sense) doesn't support fp8."""
      def grad(out_grad):
        return qdq_and_update(
            out_grad, FAKE_E5M2, self.output_grad_scale, self.
            output_grad_amax_history)
      return output, grad
  
    @tf.custom_gradient
    def kernel_qdq(self, kernel):
      """Quantize-dequantize the kernel but not its gradient."""

      qkernel  = qdq_and_update(kernel, FAKE_E4M3, self.kernel_scale, 
                                self.kernel_amax_history)
  
      def grad(kernel_grad):
        return kernel_grad
  
      return qkernel, grad

    def call(self, inputs):
        ret = tf.einsum(self.equation, self.in_qdq(inputs), 
                        self.kernel_qdq(self.kernel))
        if self.bias is not None:
            ret += self.bias
        if self.activation is not None:
            ret = self.activation(ret)
        return ret


def _analyze_einsum_string(equation, bias_axes, input_shape, output_shape):
    """Analyzes an einsum string to determine the required weight shape."""

    dot_replaced_string = re.sub(r"\.\.\.", "0", equation)

    # This is the case where no ellipses are present in the string.
    split_string = re.match(
        "([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)", dot_replaced_string
    )
    if split_string:
        return _analyze_split_string(
            split_string, bias_axes, input_shape, output_shape
        )

    # This is the case where ellipses are present on the left.
    split_string = re.match(
        "0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)", dot_replaced_string
    )
    if split_string:
        return _analyze_split_string(
            split_string, bias_axes, input_shape, output_shape, left_elided=True
        )

    # This is the case where ellipses are present on the right.
    split_string = re.match(
        "([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0", dot_replaced_string
    )
    if split_string:
        return _analyze_split_string(
            split_string, bias_axes, input_shape, output_shape
        )

    raise ValueError(
        f"Invalid einsum equation '{equation}'. Equations must be in the form "
        "[X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]...."
    )


def _analyze_split_string(
    split_string, bias_axes, input_shape, output_shape, left_elided=False
):
    """Analyze an pre-split einsum string to find the weight shape."""
    input_spec = split_string.group(1)
    weight_spec = split_string.group(2)
    output_spec = split_string.group(3)
    elided = len(input_shape) - len(input_spec)

    if isinstance(output_shape, int):
        output_shape = [output_shape]
    else:
        output_shape = list(output_shape)

    output_shape.insert(0, input_shape[0])

    if elided > 0 and left_elided:
        for i in range(1, elided):
            # We already inserted the 0th input dimension at dim 0, so we need
            # to start at location 1 here.
            output_shape.insert(1, input_shape[i])
    elif elided > 0 and not left_elided:
        for i in range(len(input_shape) - elided, len(input_shape)):
            output_shape.append(input_shape[i])

    if left_elided:
        # If we have beginning dimensions elided, we need to use negative
        # indexing to determine where in the input dimension our values are.
        input_dim_map = {
            dim: (i + elided) - len(input_shape)
            for i, dim in enumerate(input_spec)
        }
        # Because we've constructed the full output shape already, we don't need
        # to do negative indexing.
        output_dim_map = {
            dim: (i + elided) for i, dim in enumerate(output_spec)
        }
    else:
        input_dim_map = {dim: i for i, dim in enumerate(input_spec)}
        output_dim_map = {dim: i for i, dim in enumerate(output_spec)}

    for dim in input_spec:
        input_shape_at_dim = input_shape[input_dim_map[dim]]
        if dim in output_dim_map:
            output_shape_at_dim = output_shape[output_dim_map[dim]]
            if (
                output_shape_at_dim is not None
                and output_shape_at_dim != input_shape_at_dim
            ):
                raise ValueError(
                    "Input shape and output shape do not match at shared "
                    f"dimension '{dim}'. Input shape is {input_shape_at_dim}, "
                    "and output shape "
                    f"is {output_shape[output_dim_map[dim]]}."
                )

    for dim in output_spec:
        if dim not in input_spec and dim not in weight_spec:
            raise ValueError(
                f"Dimension '{dim}' was specified in the output "
                f"'{output_spec}' but has no corresponding dim in the input "
                f"spec '{input_spec}' or weight spec '{output_spec}'"
            )

    weight_shape = []
    for dim in weight_spec:
        if dim in input_dim_map:
            weight_shape.append(input_shape[input_dim_map[dim]])
        elif dim in output_dim_map:
            weight_shape.append(output_shape[output_dim_map[dim]])
        else:
            raise ValueError(
                f"Weight dimension '{dim}' did not have a match in either "
                f"the input spec '{input_spec}' or the output "
                f"spec '{output_spec}'. For this layer, the weight must "
                "be fully specified."
            )

    if bias_axes is not None:
        num_left_elided = elided if left_elided else 0
        idx_map = {
            char: output_shape[i + num_left_elided]
            for i, char in enumerate(output_spec)
        }

        for char in bias_axes:
            if char not in output_spec:
                raise ValueError(
                    f"Bias dimension '{char}' was requested, but is not part "
                    f"of the output spec '{output_spec}'"
                )

        first_bias_location = min(
            [output_spec.find(char) for char in bias_axes]
        )
        bias_output_spec = output_spec[first_bias_location:]

        bias_shape = [
            idx_map[char] if char in bias_axes else 1
            for char in bias_output_spec
        ]

        if not left_elided:
            for _ in range(elided):
                bias_shape.append(1)
    else:
        bias_shape = None

    return weight_shape, bias_shape, output_shape
