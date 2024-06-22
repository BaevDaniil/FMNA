def sortBySum(data):
    return data.sort_index(axis = 1, key = lambda x: data[x].sum(), ascending = False, inplace = True)