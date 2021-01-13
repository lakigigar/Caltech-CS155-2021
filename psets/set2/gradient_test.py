import numpy as np

def loss_test(student_loss):
    """
    Function to check if you computed the correct gradient
    
    Returns a bool, true if the loss is calculated correctly
    
    Use the following x and y as inputs into your loss function:

    x_example = np.array([[ 1.00000000e+00,  1.51762786e+00, -3.86918125e-01,
        2.28234130e-01, -1.05289605e+00,  1.87082054e+00,
        6.14515777e-01,  8.41069242e-01, -5.60342137e-01,
        1.05889717e+00,  1.08911423e+00, -5.82499016e-02,
        2.14525874e+00,  8.28253257e-01],
       [ 1.00000000e+00,  3.88424150e-01, -3.10170410e-01,
        -7.80727725e-01, -2.39455334e+00,  8.15129033e-02,
        3.24193363e-01,  4.06557837e-01, -7.49967225e-01,
        -1.01656594e+00,  2.76053483e-01,  1.99840144e-16,
        9.43303623e-01,  7.86503815e-01]])
    y_example = np.array([-1., -1.])
    w_example = np.array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
        14.])
    """
    correct_output_loss = 39.25966853098522
    return np.isclose(student_loss, correct_output_loss, atol=1e-5)

    
def update_test(student_updates):
    """
    Function to check if you computed the correct update
    
    Returns an array of bools, true if the corresponding weight's gradient is calculated
    correctly
    
    Use the following x, y, eta, and lambda as inputs into your loss function:
    
    x_example = np.array([ 1.        ,  1.51762786, -0.38691813,  0.22823413, -1.05289605,
        1.87082054,  0.61451578,  0.84106924, -0.56034214,  1.05889717,
        1.08911423, -0.0582499 ,  2.14525874,  0.82825326])
    y_example = -1.0
    eta_example = 5e-4
    lambda_example = 1e-5 * 5 ** 10
    w_example = np.array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
        14.])
    
    The update calculated should equal the loss gradient * eta
    """
    correct_output_updates = np.array([0.09815625, 0.19607131, 0.29277529, 0.39073912, 0.4877548 ,
                                        0.58687291, 0.68390101, 0.78167053, 0.87862608, 0.97709195,
                                        1.07476331, 1.17184588, 1.27060388, 1.36760163])
    return np.isclose(student_updates, correct_output_updates, atol=1e-5)