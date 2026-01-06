# In this section residual = we will calculate difference between real orbit and expected(propagated) orbit

#this is a door of anomaly detection
import numpy as np

def compute_residual(reference_position,observed_position):
    """
    reference_posiiton --> np.ndarray shape (3,N)
    observed_position --> np.darray shape (3.N)
    """
    residual_vector = observed_position - reference_position
    residual_magnitude = np.linalg.norm(residual_vector,axis=0)

    return{
        "residual_vector": residual_vector,
        "residual_magnitude": residual_magnitude
    }


    