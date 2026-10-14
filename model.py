"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    """
    Computes the numerical gradient of a scalar function f at position x using central differences.
    x can be a scalar or a NumPy array.
    """
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)

    if x.size == 0:
        return grad
    
    # Handle scalar input
    if x.ndim == 0:
        return (f(x + eps) - f(x - eps)) / (2.0 * eps)
        
    # Handle array input element-wise
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]
        
        x[idx] = old_val + eps
        pos_f = f(x.copy())
        
        x[idx] = old_val - eps
        neg_f = f(x.copy())
        
        grad[idx] = (pos_f - neg_f) / (2.0 * eps)
        x[idx] = old_val  # Reset element to original value
        it.iternext()
        
    return grad

# Step 2 - gradient_check
def gradient_check(analytical_grad, numerical_grad, tol=1e-5):
    """
    Computes the maximum coordinate-wise relative error between the analytical
    and numerical gradients, using tol as a safety floor in the denominator.
    """
    grad_a = np.asarray(analytical_grad, dtype=np.float64)
    grad_n = np.asarray(numerical_grad, dtype=np.float64)
    
    # 1. Compute element-wise absolute difference (numerator)
    abs_diff = np.abs(grad_a - grad_n)
    
    # 2. Compute element-wise maximum magnitude
    max_mag = np.maximum(np.abs(grad_a), np.abs(grad_n))
    
    # 3. Apply tol as a safety floor to the denominator
    denominator = np.maximum(max_mag, tol)
    
    # 4. Handle the edge case where both are exactly zero
    # If both are exactly 0.0, the max_mag is 0.0. If abs_diff is also 0.0, error is 0.0
    # We can mask where both are zero to output 0.0 exactly
    both_zero_mask = (grad_a == 0.0) & (grad_n == 0.0)
    
    # Calculate element-wise relative error
    rel_errors = abs_diff / denominator
    rel_errors[both_zero_mask] = 0.0
    
    # Return the maximum relative error found across all elements
    return float(np.max(rel_errors))

# Step 3 - make_dense (not yet solved)
# TODO: implement

# Step 4 - make_activation (not yet solved)
# TODO: implement

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

