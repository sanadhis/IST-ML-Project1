import numpy as np

def standardize_with_power_terms(x, power, with_ones = True, impute_with = 'mean', with_sqrt = False):
    mask = (x != -999)

    for i in range(x.shape[1]):
        x[~mask[:,i],i] = np.median(x[mask[:,i],i], axis = 0)
    
    mean = np.mean(x, axis = 0)
    std_dev = np.std(x, axis = 0)
    stand_x = (x - mean) / std_dev

    if with_sqrt:
        for i in range(x.shape[1]):
            # if all(num > 0 for num in stand_x[:,i]):
            min_ = np.min(x[:, i])
            x_sqrt = 1/(1+np.log(1 + x[:,i] - min_))
            x_sqrt = (x_sqrt - np.mean(x_sqrt))/np.std(x_sqrt)
            stand_x = np.concatenate((stand_x,x_sqrt.reshape([x.shape[0],1])),1)
            x_sqrt = np.log(1 + x[:,i] - min_)
            x_sqrt = (x_sqrt - np.mean(x_sqrt))/np.std(x_sqrt)
            stand_x = np.concatenate((stand_x,x_sqrt.reshape([x.shape[0],1])),1)

    x_tmp = x
    for deg in range(2, power + 1):
        x_tmp = x_tmp * x
        x_sq = x_tmp

        mean = (x_sq * mask).sum(axis=0)/np.sum(mask, axis=0)
        std_dev = np.sqrt((((x_sq - mean) * mask)**2).sum(axis=0)/np.sum(mask, axis=0))

        x_sq = (x_sq * mask - mean)/std_dev
        x_sq[~mask] = 0
        stand_x = np.concatenate((stand_x,x_sq),axis = 1)


    # ---------- Add ones to the matrix --------
    if with_ones:
        tmp = np.ones([stand_x.shape[0], stand_x.shape[1] + 1])
        tmp[:,1:] = stand_x
        stand_x = tmp

    return stand_x
