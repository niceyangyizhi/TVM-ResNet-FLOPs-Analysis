import functools
import operator

# Calculate FLOPs for nn.conv2d operator
def count_flops_nn_conv2d(expr):
    attrs = expr.attrs
    args = expr.args
    input_shape = args[0].checked_type.shape
    batch_size, in_channel, in_height, in_width = [int(x) for x in input_shape]
    kernel_shape = args[1].checked_type.shape
    out_channel, _, kernel_height, kernel_width = [int(x) for x in kernel_shape]
    groups = attrs.groups
    strides = attrs.strides
    stride_height, stride_width = [int(x) for x in strides]
    padding = attrs.padding
    pad_top, pad_left, pad_bottom, pad_right = [int(x) for x in padding]
    dilation = attrs.dilation
    dilation_height, dilation_width = [int(x) for x in dilation]
    out_height = (in_height + pad_top + pad_bottom - (dilation_height * (kernel_height - 1) + 1)) // stride_height + 1
    out_width = (in_width + pad_left + pad_right - (dilation_width * (kernel_width - 1) + 1)) // stride_width + 1
    flops = 2 * batch_size * out_channel * in_channel * kernel_height * kernel_width * out_height * out_width // groups
    return flops



# Calculate FLOPs for nn.batch_norm operator
def count_flops_nn_batch_norm(expr):
    args = expr.args
    input_shape = args[0].checked_type.shape
    batch_size, in_channel, in_height, in_width = input_shape
    flops = 4 * batch_size * in_channel * in_height * in_width
    return flops

# Calculate FLOPs for nn.relu operator
def count_flops_nn_relu(expr):
    args = expr.args
    input_shape = args[0].checked_type.shape
    batch_size, in_channel, in_height, in_width = input_shape
    flops = batch_size * in_channel * in_height * in_width
    return flops

# Calculate FLOPs for add operator
def count_flops_add(expr):
    args = expr.args
    input_shape = args[0].checked_type.shape
    flops = functools.reduce(operator.mul, input_shape)
    return flops


# Calculate FLOPs for nn.max_pool2d operator
def count_flops_nn_max_pool2d(expr):
    return 0

# Calculate FLOPs for nn.global_avg_pool2d operator
def count_flops_nn_global_avg_pool2d(expr):
    return 0

# Calculate FLOPs for nn.batch_flatten operator
def count_flops_nn_batch_flatten(expr):
    return 0

# Calculate FLOPs for nn.dense operator
def count_flops_nn_dense(expr):
    args = expr.args
    input_shape = args[0].checked_type.shape
    weight_shape = args[1].checked_type.shape
    units, _ = weight_shape
    flops = functools.reduce(operator.mul, input_shape) * units
    return flops


count_flops_op_map = {
    "nn.conv2d": count_flops_nn_conv2d,
    "nn.batch_norm": count_flops_nn_batch_norm,
    "nn.relu": count_flops_nn_relu,
    "add": count_flops_add,
    "nn.max_pool2d": count_flops_nn_max_pool2d,
    "nn.global_avg_pool2d": count_flops_nn_global_avg_pool2d,
    "nn.batch_flatten": count_flops_nn_batch_flatten,
    "nn.dense": count_flops_nn_dense
}