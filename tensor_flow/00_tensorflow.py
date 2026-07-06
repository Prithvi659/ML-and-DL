import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

import tensorflow as tf
# print(tf.__version__)

# create a scalar
scalar = tf.constant(7)
# print(scalar)
# print(scalar.ndim) # no of dimension

# create a vector
vector = tf.constant([10,10])
# print(vector)
# print(vector.ndim)

# create a matrix
matrix = tf.constant([[10,10],[10,10]])
# print(matrix)
# print(matrix.ndim) 

another_matrix = tf.constant([[19,16],
                              [21,17],
                              [18,19]],dtype=tf.float16)
# print(another_matrix)
# print(another_matrix.ndim)

# create a tensor
tensor = tf.constant([
    [
        [1,2,3],[1,2,3],[1,2,3]
    ],
    [
        [4,5,6],[4,5,6],[4,5,6]
    ],
    [
        [7,8,9],[7,8,9],[7,8,9]
    ]
])
# print(tensor)
# print(tensor.ndim)

# same tensor with variable fuc
changeable_tensor = tf.Variable([10,7])
unchangeable_tensor = tf.constant([10,7])
# print(changeable_tensor,unchangeable_tensor)

changeable_tensor.scatter_nd_update(
    indices=[[0]],
    updates=[7]
)
# print(changeable_tensor)

# randomly suffeling the tensor values does not suffle the inner values
unsuffeled = tf.constant([
    [[1,2,3],[1,2,3],[1,2,3]],
    [[4,5,6],[4,5,6],[4,5,6]],
    [[7,8,9],[7,8,9],[7,8,9]]
    ])
# print(unsuffeled,unsuffeled.ndim) 
tf.random.set_seed(42) # globally assigned
suffled = tf.random.shuffle(unsuffeled,seed=42)
# print(suffled)

# create a tensr of ones and zeros
# print(tf.ones([3,2],dtype="int32"),"\n",tf.zeros([2,3],dtype="int32"))

# selecting elements from big tensor
ones = tf.ones([2,3,4,5],dtype=tf.int32)
# print(ones[:1,:1,:1,:],"\n",ones[:1,:1,:,:1],"\n",ones[:1,:,:1,1:],"\n",ones[:,:1,:1,:1])

# adding a new dim to a created matrix before shape=(2, 2)
# print(matrix,tf.expand_dims(matrix,axis=-1))  # shape=(2, 2, 1)  after axis ca be set anywhere start ,end, middle

# matrix multipication should use reshape/transpose for this because inner dim should match
matrix_a  = tf.constant([[2,4,3],
                         [4,3,2],
                         [4,3,2]])
matrix_b = tf.constant([[8,2,3],
                        [6,4,2]])
# print(f"before : {matrix_a}\n,{matrix_b} \n ,after : {tf.matmul(matrix_a,tf.transpose(matrix_b))}")
# print(tf.tensordot(tf.transpose(matrix_a),matrix_b),axis = 1)

print(tf.config.list_physical_devices('GPU'))