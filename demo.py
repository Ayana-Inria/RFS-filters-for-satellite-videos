from src.filter_fns import track_with_RFS_filter
#from src.CNNs.cnn_only_filter import track_with_CNN_filter

import src.data_utils.data_reader as data_reader
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    track_with_RFS_filter()
    #track_with_CNN_filter()

if __name__ == '__main__':
    print("Starting Demo")
    main()

