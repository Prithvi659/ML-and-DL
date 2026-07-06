import numpy as np
import math

# %%
print(np.__version__)

# arr = np.array([1,2,3,4])
# print(arr)

lis = arr.tolist()
print(lis)

# 2D arry
# arr = np.array([[1,2,3],
#                 [4,5,6],
#                 [7,8,9]])
# print(arr[1,2])

# 3D array
# arr = np.array([[['1','2','3'],['4','5','6'],['7','8','9']],
#                  [['10','11','12'],['13','14','15'],['16','17','18']],
#                  [['19','20','21'],['22','23','24'],['25','26',' ' ]]])

# print(arr[0,0,0]+arr[0,2,1]+arr[0,1,0]+arr[0,0,1]+arr[0,0,2]) # string concatation

# slicing [start:end:step]
# arr = np.array([[1,2,3,4],
                # [5,6,7,8],
                # [9,10,11,12],
                # [13,14,15,16]])

# row selection
# print(arr[1::2])
#  
# coloum selection
#
# mix
# print(arr[0:4:2,0:4:2])


# square function
# arr = np.array([1,-3,15,-446])
# print(np.square(arr))
# print(np.sqrt(arr))
# print(np.cbrt(arr))

#sin() function
# arr = [0,math.pi/2,math.pi/3,math.pi/6]
# sin_values = np.sin(arr)
# cos_values = np.cos(arr)
# cosh_values = np.cosh(arr) # cosh , h = means for hyperbolic functions
# tan_values = np.tan(arr)
# radian_2_degree = np.rad2deg(arr)
# degree_2_radian = np.deg2rad(arr)
# print(f"sin values: {np.round(sin_values,4)}\n cos values: {np.round(cos_values,4)}\n cosh values: {np.round(cosh_values,4)}\n"
#       f"tan values: {np.round(tan_values,4)}\n radian to degree values: {radian_2_degree}\n"
#       f"degree to radian values: {degree_2_radian}")

# power function

# arr_1 = [2,2,2,2,2]
# arr_2 = [2,3,4,5,6]
# final = np.power(arr_1,arr_2)
# print(final)

# absolute function
# arr = [1,-3,15,-466]
# print(np.absolute(arr))
# a = 3 + 5j
# b = 2 + 4j
# print(np.absolute(a))
# print(np.abs(b))

#log and exp functions
# arr = [4,5,6]
# log_values = np.log(arr)
# exp_values = np.exp(arr)
# print(log_values,"\n",exp_values)