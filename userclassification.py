import numpy as np


def user_classification():

    computing_power = np.random.rand(100)
    datarate = np.random.rand(100)
    resource = []
    for i in range(100):
        a = computing_power[i] + datarate[i]
        resource.append(a)
    
    
    user_class = []
    for i in range(len(resource)):
        
        if 0 <= resource[i]< 2/6:
            user_class.append(1)
        if 2/6 <= resource[i]< 4/6:
            user_class.append(2)
        if 4/6 <= resource[i]< 6/6:
            user_class.append(3)
        if 6/6 <= resource[i]< 8/6:
            user_class.append(4)
        if 8/6 <= resource[i]< 10/6:
            user_class.append(5)
        if 10/6 <= resource[i]< 2:
            user_class.append(6)
  
    return user_class

# uc = user_classification()
# print(uc)
# print(uc.count(0))
# print(uc.count(1))
# print(uc.count(2))
# print(uc.count(3))
# print(uc.count(4))
# print(uc.count(5))