import numpy as np

def get_major_axis(mask):
    from skimage.measure import label, regionprops
    
    # divide entire masks into each instance using connected-components labelling
    labels = label(mask)
    
    # iterate to calculate the length of the major axis of each instance
    major_axis_list = [regionprops((labels == i).astype('uint8'))[0].major_axis_length \
                       for i in np.unique(labels) if i != 0]
    
    # find the longest major axis
    longest_major_axis = max(major_axis_list)
    return longest_major_axis