'''
utilties for fuzz testing

Stanley Bak
'''

import pickle

def total_size(obj):
    'get pickled size of object'

    return len(pickle.dumps(obj, -1))
