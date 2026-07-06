import numpy as np

# nums = np.array(range(31))
# nums  = np.arange(100,151,5)
# print(nums)

# ones function
# nums = np.ones((3,2,2),int)
# print(nums)

# nums = np.zeros((3,3),int)
# print(nums)

#identity matrix
# nums = np.eye(4,3,1,int)
# print(nums)

# full matrix crates all the elements of same value
# nums = np.full((3,2,2),5,int)
# print(nums)

# linspace function is used to generate linearly space values the difference will be same between them
nums =np.linspace(1,1000,10,retstep=True)
print(nums)