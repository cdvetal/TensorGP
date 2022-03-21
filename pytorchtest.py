import torch
import math
import numpy as np
import tensorflow as tf

_min_domain = -1
_max_domain = 1
_domain_delta = _max_domain - _min_domain
dtype = torch.float32
#dev = torch.device("cpu")
dev = torch.device("cuda:0")
res = [8, 8]

# Test CUDA
print("Is CUDA available:\t", torch.cuda.is_available())
cur_dev = torch.cuda.current_device()
print("Current CUDA device:", dev)
print("CUDA device name:\t", torch.cuda.get_device_name(cur_dev))
print("\n")
torch.cuda.device(cur_dev)

## ====================== tensorflow operators ====================== ##
def resolve_var_node(dimensions, n):
    temp = np.ones(len(dimensions), dtype=int)
    temp[n] = -1
    res = tf.reshape(tf.range(dimensions[n], dtype=tf.float32), temp)
    _resolution = dimensions[n]
    dimensions[n] = 1
    res = tf.scalar_mul(((1.0 / (_resolution - 1)) * _domain_delta), res)
    res = tf.math.add(res, tf.constant(_min_domain, tf.float32, res.shape))
    res = tf.tile(res, dimensions)
    return res

def resolve_abs_node(child1, dims=[]):
    return tf.math.abs(child1)

def resolve_add_node(child1, child2, dims=[]):
    return tf.math.add(child1, child2)

def resolve_and_node(child1, child2, dims=[]):
    left_child_tensor = tf.cast(tf.scalar_mul(1e6, child1), tf.int32)
    right_child_tensor = tf.cast(tf.scalar_mul(1e6, child2), tf.int32)
    return tf.scalar_mul(1e-6, tf.cast(tf.bitwise.bitwise_and(left_child_tensor, right_child_tensor), tf.float32))

def resolve_xor_node(child1, child2, dims=[]):
    left_child_tensor = tf.cast(tf.scalar_mul(1e6, child1), tf.int32)
    right_child_tensor = tf.cast(tf.scalar_mul(1e6, child2), tf.int32)
    return tf.scalar_mul(1e-6, tf.cast(tf.bitwise.bitwise_xor(left_child_tensor, right_child_tensor), tf.float32))

def resolve_or_node(child1, child2, dims=[]):
    left_child_tensor = tf.cast(tf.scalar_mul(1e6, child1), tf.int32)
    right_child_tensor = tf.cast(tf.scalar_mul(1e6, child2), tf.int32)
    return tf.scalar_mul(1e-6, tf.cast(tf.bitwise.bitwise_or(left_child_tensor, right_child_tensor), tf.float32))

def resolve_cos_node(child1, dims=[]):
    return tf.math.cos(tf.scalar_mul(math.pi, child1))

def resolve_div_node(child1, child2, dims=[]):
    left_child_tensor = tf.cast(child1, tf.float32)
    right_child_tensor = tf.cast(child2, tf.float32)
    return tf.math.divide_no_nan(left_child_tensor, right_child_tensor)

def resolve_exp_node(child1, dims=[]):
    return tf.math.exp(child1)

def resolve_if_node(child1, child2, child3, dims=[]):
    return tf.where(child3 < 0, child1, child2)

def resolve_log_node(child1, dims=[]):
    res = tf.where(child1 > 0, tf.math.log(child1), tf.constant(-1, tf.float32, dims))
    return res

def resolve_max_node(child1, child2, dims=[]):
    return tf.math.maximum(child1, child2)

def resolve_mdist_node(child1, child2, dims=[]):
    return tf.scalar_mul(0.5, tf.add(child1, child2))

def resolve_min_node(child1, child2, dims=[]):
    return tf.math.minimum(child1, child2)

def resolve_mod_node(child1, child2, dims=[]):
    return tf.math.mod(child1, child2)

def resolve_mult_node(child1, child2, dims=[]):
    return tf.math.multiply(child1, child2)

def resolve_neg_node(child1, dims=[]):
    return tf.math.negative(child1)


def resolve_pow_node(child1, child2, dims=[]):
    return tf.where(child1 == 0, tf.constant(0, tf.float32, dims), tf.math.pow(tf.math.abs(child1), tf.math.abs(child2)))

def resolve_sign_node(child1, dims=[]):
    #return tf.math.divide_no_nan(tf.math.abs(child1),child1)
    return tf.math.sign(child1)

def resolve_sin_node(child1, dims=[]):
    return tf.math.sin(tf.scalar_mul(math.pi, child1))

def resolve_sqrt_node(child1, dims=[]):
    return tf.where(child1 > 0, tf.math.sqrt(child1), tf.constant(0, tf.float32, dims))

def resolve_sub_node(child1, child2, dims=[]):
    return tf.math.subtract(child1, child2)

def resolve_tan_node(child1, dims=[]):
    return tf.where(child1 != (math.pi * 0.5), tf.math.tan(tf.scalar_mul(math.pi, child1)), tf.constant(0, tf.float32, dims))

# return x * x * x * (x * (x * 6 - 15) + 10);
# simplified to x2*(6x3 - (15x2 - 10x)) to minimize TF operations
def resolve_sstepp_node(child1, dims=[]):
    x = resolve_clamp(child1)
    x2 = tf.square(x)
    return tf.add(tf.multiply(x2,
                              tf.subtract(tf.scalar_mul(6.0 * _domain_delta, tf.multiply(x, x2)),
                                          tf.subtract(tf.scalar_mul(15.0 * _domain_delta, x2),
                                                      tf.scalar_mul(10.0 * _domain_delta, x)))),
                  tf.constant(_min_domain, dtype=tf.float32, shape=dims))

# return x * x * (3 - 2 * x);
# simplified to (6x2 - 4x3) to minimize TF operations
def resolve_sstep_node(child1, dims=[]):
    x = resolve_clamp(child1)
    x2 = tf.square(x)
    return tf.add(tf.subtract(tf.scalar_mul(3.0 * _domain_delta, x2),
                              tf.scalar_mul(2.0 * _domain_delta, tf.multiply(x2, x))),
                  tf.constant(_min_domain, dtype=tf.float32, shape=dims))

def resolve_step_node(child1, dims=[]):
    return tf.where(child1 < 0.0,
                    tf.constant(-1.0, dtype=tf.float32, shape=dims),
                    tf.constant(1.0, dtype=tf.float32, shape=dims))

def resolve_clamp(tensor):
    return tf.clip_by_value(tensor, clip_value_min=_min_domain, clip_value_max=_max_domain)
    #return tf.cast(tf.cast(tensor, tf.uint8), tf.float32)

def resolve_frac_node(child1, dims=[]):
    return tf.subtract(child1, tf.floor(child1))

def resolve_clip_node(child1, child2, child3, dims=[]): # a < n < b
    return tf.minimum(tf.maximum(child1, child2), child3)

def resolve_len_node(child1, child2, dims=[]):
    return tf.sqrt(tf.add(tf.square(child1), tf.square(child2)))

def resolve_lerp_node(child1, child2, child3, dims=[]):
    child3 = resolve_frac_node(child3, dims)
    t_dist = tf.subtract(child2, child1)
    t_dist = tf.multiply(child3, t_dist)
    return tf.math.add(child1, t_dist)

def old_tf_rmse(child1, child2):
    child1 = tf.cast(child1, tf.float32)
    child2 = tf.cast(child2, tf.float32)
    elements = np.prod(child1.shape.as_list())
    sdiff = tf.math.squared_difference(child1, child2)
    mse = tf.math.reduce_sum(sdiff).numpy() / elements
    mse = math.sqrt(mse)
    return mse

@tf.function
def tf_rmse(child1, child2):
    child1 = tf.scalar_mul(1/127.5, child1)
    child2 = tf.scalar_mul(1/127.5, child2)
    return tf.sqrt(tf.reduce_mean(tf.square(child1 - child2)))

def resolve_stack_node(nums, dimensions, edims):
    return tf.stack([tf.constant(float(carvar), tf.float32, dimensions[:edims]) for carvar in nums], axis = edims)

def final_transform_domain(final_tensor, res):
    final_tensor = tf.where(tf.math.is_nan(final_tensor), 0.0, final_tensor)

    final_tensor = tf.clip_by_value(final_tensor, clip_value_min=_min_domain, clip_value_max=_max_domain)
    final_tensor = tf.math.subtract(final_tensor, tf.constant(_min_domain, tf.float32, res))
    final_tensor = tf.scalar_mul(255 / _domain_delta, final_tensor)

    return final_tensor


#===================================================================================================

def node_var(dimensions, n):
    temp = np.ones(len(dimensions), dtype=int)
    temp[n] = -1
    res = torch.reshape(torch.arange(0, dimensions[n], dtype=dtype), tuple(temp))
    resolution = dimensions[n]
    dimensions[n] = 1
    res = torch.add(torch.full(res.shape, _min_domain), res, alpha=((1.0 / (resolution - 1)) * _domain_delta))
    return res.repeat(tuple(dimensions))

def node_abs(x1, dims=[]):
    return torch.abs(x1)

def node_add(x1, x2, dims=[]):
    return torch.add(x1, x2)

def node_sub(x1, x2, dims=[]):
    return torch.sub(x1, x2)

def node_mul(x1, x2, dims=[]):
    return torch.mul(x1, x2)

def node_div(x1, x2, dims=[]):
    return torch.where(x2 != 0.0, torch.div(x1, x2), torch.tensor(0.0, dtype=dtype, device=cur_dev))

def node_bit_and(x1, x2, dims=[]):
    left_child = torch.mul(x1, 1e6).int()
    right_child = torch.mul(x2, 1e6).int()
    return torch.mul(torch.bitwise_and(left_child, right_child), 1e-6)

def node_bit_or(x1, x2, dims=[]):
    left_child = torch.mul(x1, 1e6).int()
    right_child = torch.mul(x2, 1e6).int()
    return torch.mul(torch.bitwise_or(left_child, right_child), 1e-6)

def node_bit_xor(x1, x2, dims=[]):
    left_child = torch.mul(x1, 1e6).int()
    right_child = torch.mul(x2, 1e6).int()
    return torch.mul(torch.bitwise_xor(left_child, right_child), 1e-6)

def node_cos(x1, dims=[]):
    return torch.cos(torch.mul(x1, math.pi))

def node_sin(x1, dims=[]):
    return torch.sin(torch.mul(x1, math.pi))

def node_tan(x1, dims=[]):
    return torch.tan(torch.mul(x1, math.pi))

def node_exp(x1, dims=[]):
    return torch.exp(x1)

def node_if(x1, x2, x3, dims=[]):
    print(x3)
    return torch.where(x3 < 0, x1, x2)

def node_log(x1, dims=[]):
    return torch.where(x1 > 0.0, torch.log(x1), torch.tensor(-1.0, dtype=dtype, device=cur_dev))

def node_max(x1, x2, dims=[]):
    return torch.max(x1, x2)

def node_min(x1, x2, dims=[]):
    return torch.min(x1, x2)

def node_mdist(x1, x2, dims=[]):
    return torch.mul(torch.add(x1, x2), 0.5)

def node_mod(x1, x2, dims=[]):
    return torch.where(x2 != 0.0, torch.fmod(x1, x2), torch.tensor(0.0, dtype=dtype, device=cur_dev))
    #return torch.fmod(x1, x2)

def node_neg(x1, dims=[]):
    return torch.neg(x1)

def node_pow(x1, x2, dims=[]):
    return torch.where(x1 != 0, torch.pow(torch.abs(x1), torch.abs(x2)), torch.tensor(0.0, dtype=dtype, device=cur_dev))


def node_sign(x1, dims=[]):
    return torch.sign(x1)

def node_sqrt(child1, dims=[]):
    return torch.where(child1 > 0, torch.sqrt(child1), torch.tensor(0.0, dtype=dtype, device=cur_dev))

def tensor_rmse(x1, x2):
    x1 = torch.mul(x1, 1/127.5)
    x2 = torch.mul(x2, 1/127.5)
    return torch.sqrt(torch.mean(torch.square(torch.sub(x1, x2))))

def node_clamp(tensor):
    return torch.clip(tensor, _min_domain, _max_domain)

# return x * x * x * (x * (x * 6 - 15) + 10);
# simplified to x2*(6x3 - (15x2 - 10x)) to minimize operations
def node_sstepp(x1, dims=[]):
    x = node_clamp(x1)
    x2 = torch.square(x)

    return torch.add(torch.mul(x2,
                              torch.sub(torch.mul(torch.mul(x, x2), 6.0 * _domain_delta),
                                        torch.sub(torch.mul(x2, 15.0 * _domain_delta),
                                                  torch.mul(x, 10.0 * _domain_delta)))),
                     _min_domain)

# return x * x * (3 - 2 * x);
# simplified to (6x2 - 4x3) to minimize TF operations
def node_sstep(x1, dims=[]):
    x = node_clamp(x1)
    x2 = torch.square(x)
    return torch.add(torch.sub(torch.mul(3.0 * _domain_delta, x2),
                               torch.mul(2.0 * _domain_delta, torch.mul(x2, x))),
                  torch.tensor(_min_domain, dtype=dtype, device=cur_dev))

def node_step(x1, dims=[]):
    return torch.where(x1 < 0.0,
                    torch.tensor(-1.0, dtype=dtype, device=cur_dev),
                    torch.tensor(1.0, dtype=dtype, device=cur_dev))

def node_frac(x1, dims=[]):
    return torch.frac(x1)

def node_len(x1, x2, dims=[]):
    return torch.hypot(x1, x2)

def node_lerp(x1, x2, x3, dims=[]):
    return torch.lerp(x1, x2, node_frac(node_abs(x3)))

def node_stack(nums, dimensions, edims):
    return torch.stack([torch.full(dimensions[:edims], float(carvar), dtype=dtype, device=cur_dev) for carvar in nums], dim = edims)

def final_transform_domain1(final_tensor):
    final_tensor = torch.where(torch.isnan(final_tensor), torch.tensor(0.0, dtype=dtype, device=cur_dev), final_tensor)

    final_tensor = node_clamp(final_tensor)
    final_tensor = torch.sub(final_tensor, _min_domain)
    final_tensor = torch.mul(final_tensor, 255 / _domain_delta)

    return final_tensor

#fset vars tests (only torch)
var_x = node_var(np.copy(res), 0)
var_y = node_var(np.copy(res), 1)

#print("Var x: ", var_x)
#print("Var y: ", var_y)

torch_tests = []
tf_tests = []
torch_vars = []
tf_vars = []



torch_vars.append(torch.tensor(np.array([[-1, 2.2, 3], [0, 5.5, -6.9]]), dtype=dtype, device=cur_dev))
torch_vars.append(torch.tensor(np.array([[1, 2, 3], [4, 5, 0]]), dtype=dtype, device=cur_dev))
torch_vars.append(torch.tensor(np.array([[1, 1, 0], [1, 0, 1]]), dtype=dtype, device=cur_dev))
torch_vars.append(torch.tensor(np.array([[0.3, 0.5, 1.2], [0.2, 0.7, 0.6]]), dtype=dtype, device=cur_dev))
torch_vars.append(torch.tensor(np.array([[-2, 3], [4.5, float('-inf')], [0, float('nan')]]), dtype=dtype, device=cur_dev))


tf_vars.append(tf.convert_to_tensor(np.array([[-1, 2.2, 3], [0, 5.5, -6.9]], dtype=np.float32)))
tf_vars.append(tf.convert_to_tensor(np.array([[1, 2, 3], [4, 5, 0]], dtype=np.float32)))
tf_vars.append(tf.convert_to_tensor(np.array([[1, 1, 0], [1, 0, 1]], dtype=np.float32)))
tf_vars.append(tf.convert_to_tensor(np.array([[0.3, 0.5, 1.2], [0.2, 0.7, 0.6]], dtype=np.float32)))
tf_vars.append(tf.convert_to_tensor(np.array([[-2, 3], [4.5, float('-inf')], [0, float('nan')]], dtype=np.float32)))


# Test variable assert
print("Asserting initial variables")
assert len(torch_vars) == len(tf_vars)
for i in range(len(tf_vars)):
    if not np.array_equal(torch_vars[i], tf_vars[i]):
        print("Error in init var: ", i)


# Test operators (torch)
print("\nRunning torch tests: \n")
a = torch_vars[0]; print("Torch a: ", a.data)
b = torch_vars[1]; print("Torch b: ", b.data)
c = torch_vars[2]; print("Torch c: ", c.data)
w = torch_vars[3]; print("Torch w: ", w.data)
t = torch_vars[4]; print("Torch w: ", t.data)

cnt = 0
for i in torch_vars:
    print("Type of torch var", cnt, ": ", i.device)
    print(i)
    cnt += 1


tmp = node_abs(a); torch_tests.append(tmp); print(tmp.data)
tmp = node_add(a, b); torch_tests.append(tmp); print(tmp.data)
tmp = node_sub(a, b); torch_tests.append(tmp); print(tmp.data)
tmp = node_mul(a, b); torch_tests.append(tmp); print(tmp.data)
tmp = node_div(a, b); torch_tests.append(tmp); print(tmp.data)
tmp = node_bit_and(a, b); torch_tests.append(tmp); print(tmp.data)
tmp = node_bit_or(a, b); torch_tests.append(tmp); print(tmp.data)
tmp = node_bit_xor(a, b); torch_tests.append(tmp); print(tmp.data)
tmp = node_cos(a); torch_tests.append(tmp); print(tmp.data) ##
tmp = node_sin(a); torch_tests.append(tmp); print(tmp.data) ##
tmp = node_tan(a); torch_tests.append(tmp); print(tmp.data) ##
tmp = node_if(a, b, c); torch_tests.append(tmp); print(tmp.data)
tmp = node_exp(a); torch_tests.append(tmp); print(tmp.data) ##
tmp = node_log(a); torch_tests.append(tmp); print(tmp.data)
tmp = node_max(a, b); torch_tests.append(tmp); print(tmp.data)
tmp = node_min(a, b); torch_tests.append(tmp); print(tmp.data)
tmp = node_mdist(a, b); torch_tests.append(tmp); print(tmp.data)
tmp = node_mod(a, b); torch_tests.append(tmp); print(tmp.data) ##
tmp = node_neg(a); torch_tests.append(tmp); print(tmp.data)
tmp = node_pow(a, b); torch_tests.append(tmp); print(tmp.data)
tmp = node_sign(a); torch_tests.append(tmp); print(tmp.data)
tmp = node_sqrt(a); torch_tests.append(tmp); print(tmp.data)
tmp = tensor_rmse(a, b); torch_tests.append(tmp); print(tmp.data)
tmp = node_clamp(a); torch_tests.append(tmp); print(tmp.data)
tmp = node_sstepp(a); torch_tests.append(tmp); print(tmp.data)
tmp = node_sstep(a); torch_tests.append(tmp); print(tmp.data)
tmp = node_step(a); torch_tests.append(tmp); print(tmp.data)
tmp = node_frac(a); torch_tests.append(tmp); print(tmp.data) ##
tmp = node_len(a, b); torch_tests.append(tmp); print(tmp.data)
tmp = node_lerp(a, b, w); torch_tests.append(tmp); print(tmp.data)
#tmp = node_stack([3, 7, 1], [8, 8, 3], 2); torch_tests.append(tmp); print(tmp.data)
tmp = final_transform_domain1(t); torch_tests.append(tmp); print(tmp.data)
print("temporary var: ", tmp.device)


# Test operators (TF)
print("\nRunning tf tests: \n")
a = tf_vars[0]; print("TF a: ", a.numpy())
b = tf_vars[1]; print("TF b: ", b.numpy())
c = tf_vars[2]; print("TF c: ", c.numpy())
w = tf_vars[3]; print("TF w: ", w.numpy())
t = tf_vars[4]; print("TF w: ", t.numpy())

tmp = resolve_abs_node(a); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_add_node(a, b); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_sub_node(a, b); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_mult_node(a, b); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_div_node(a, b); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_and_node(a, b); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_or_node(a, b); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_xor_node(a, b); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_cos_node(a); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_sin_node(a); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_tan_node(a); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_if_node(a, b, c); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_exp_node(a); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_log_node(a); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_max_node(a, b); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_min_node(a, b); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_mdist_node(a, b); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_mod_node(a, b); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_neg_node(a); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_pow_node(a, b); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_sign_node(a); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_sqrt_node(a); tf_tests.append(tmp); print(tmp.numpy())
tmp = tf_rmse(a, b); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_clamp(a); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_sstepp_node(a); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_sstep_node(a); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_step_node(a); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_frac_node(a); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_len_node(a, b); tf_tests.append(tmp); print(tmp.numpy())
tmp = resolve_lerp_node(a, b, w); tf_tests.append(tmp); print(tmp.numpy())
#tmp = resolve_stack_node([3, 7, 1], [8, 8, 3], 2); tf_tests.append(tmp); print(tmp.numpy())
tmp = final_transform_domain(t, (3, 2)); tf_tests.append(tmp); print(tmp.numpy())

# the fraction node in TF is not prepared for negative numbers

# Test operator assert
print("Asserting operator arrays")
assert len(tf_tests) == len(torch_tests)
for i in range(len(tf_tests)):
    if not np.array_equal(torch_tests[i].data.cpu().numpy(), tf_tests[i].numpy()):
        print("\nError in tests case: ", i)
        print("Torch res\n", torch_tests[i].data)
        print("TF res\n", tf_tests[i].numpy())






