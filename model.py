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

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
    """
    Creates a dictionary representing a dense (linear) network layer matching the precise API contract.
    
    CONTRACT:
        forward(x) -> (y, cache)
        backward(dout, cache) -> (dx, grads)
    """
    # Initialize the parameters using the provided function
    W, b = weight_init_fn(in_dim, out_dim)
    params = {'W': W, 'b': b}
    
    def forward(x):
        # Affine map: y = x @ W + b
        y = np.dot(x, params['W']) + params['b']
        # Store x in the cache to pass to backward
        cache = x
        return y, cache
        
    def backward(dout, cache):
        x = cache
        
        # dL/dW = x.T @ dout
        dW = np.dot(x.T, dout)
        # dL/db = sum over the batch axis (axis 0)
        db = np.sum(dout, axis=0)
        # dL/dx = dout @ W.T
        dx = np.dot(dout, params['W'].T)
        
        grads = {'W': dW, 'b': db}
        return dx, grads
        
    return {
        'params': params,
        'forward': forward,
        'backward': backward
    }

# Step 4 - make_activation
def make_activation(kind="relu"):
    """
    Creates an activation layer matching the protocol contract.
    Supports "relu" and "tanh".
    
    CONTRACT:
        forward(x) -> (y, cache)
        backward(dout, cache) -> (dx, grads)
    """
    if kind not in ("relu", "tanh"):
        raise ValueError(f"Unsupported activation type: {kind}")
        
    def forward(x):
        if kind == "relu":
            # For ReLU, cache the input x to construct the activation mask later
            cache = x
            y = np.maximum(0, x)
        elif kind == "tanh":
            # For Tanh, computing the derivative is faster if we cache the output y
            y = np.tanh(x)
            cache = y
        return y, cache
        
    def backward(dout, cache):
        grads = {} # Activation layers contain no learnable parameter arrays
        
        if kind == "relu":
            x = cache
            dx = dout * (x > 0)
        elif kind == "tanh":
            y = cache
            # dL/dx = dout * (1 - y^2)
            dx = dout * (1.0 - y ** 2)
            
        return dx, grads
        
    return {
        'params': {},
        'forward': forward,
        'backward': backward
    }

# Step 5 - initialize_weights
def initialize_weights(in_dim, out_dim, scheme="he"):
    """
    Initializes weights W and biases b using the requested scaling metric.
    Returns:
        W: array of shape (in_dim, out_dim)
        b: array of shape (out_dim,)
    """
    if scheme == "he":
        std = np.sqrt(2.0 / in_dim)
    elif scheme == "xavier":
        std = np.sqrt(2.0 / (in_dim + out_dim))
    else:
        raise ValueError(f"Unknown initialization scheme: {scheme}")
        
    # Generate weights from normal distribution and scale by standard deviation
    W = np.random.randn(in_dim, out_dim) * std
    # Initialize biases to zeros
    b = np.zeros(out_dim)
    
    return W, b

# Step 6 - make_loss
def make_loss(kind="cross_entropy"):
    """
    Creates a loss objective matching the loss layer contract.
    CONTRACT:
        loss_fn(score_logits, y) -> (loss_scalar, dscores)
    """
    if kind != "cross_entropy":
        raise ValueError(f"Unsupported loss type: {kind}")
        
    def loss_fn(score_logits, y):
        # 1. Softmax computation with numerical stability trick
        shifted_logits = score_logits - np.max(score_logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # 2. Compute scalar cross-entropy loss over the batch size
        num_examples = score_logits.shape[0]
        core_loss = -np.log(probs[np.arange(num_examples), y] + 1e-15)
        loss_scalar = np.sum(core_loss) / num_examples
        
        # 3. Compute gradient with respect to logits (dscores)
        dscores = probs.copy()
        dscores[np.arange(num_examples), y] -= 1.0
        dscores /= num_examples
        
        return float(loss_scalar), dscores
        
    return loss_fn

# Step 7 - make_sequential
def make_sequential(layers):
    """
    Compose protocol-honoring layers into one sequential model.
    
    CONTRACT:
        forward(x) -> (y, caches)
        backward(dout, caches) -> (dx, grads_list)
        params: list of dicts referencing each layer's live parameter arrays in order.
    """
    # Aggregate a live list view of all individual layer parameters
    params = [layer['params'] for layer in layers]
    
    def forward(x):
        caches = []
        current_activation = x
        
        # Route forward activations left-to-right
        for layer in layers:
            current_activation, cache = layer['forward'](current_activation)
            caches.append(cache)
            
        return current_activation, caches
        
    def backward(dout, caches):
        # Pre-allocate grads_list to preserve structural alignment with the input layers array
        grads_list = [None] * len(layers)
        current_dout = dout
        
        # Route backward gradients right-to-left
        for idx in reversed(range(len(layers))):
            layer = layers[idx]
            cache = caches[idx]
            
            # Backpropagate through individual layer
            dx, grads_dict = layer['backward'](current_dout, cache)
            
            # Store the computed layer gradients into the matching index slot
            grads_list[idx] = grads_dict
            
            # Downstream gradient becomes upstream gradient for the next layer down
            current_dout = dx
            
        return current_dout, grads_list
        
    return {
        'params': params,
        'layers': layers, # Preserved for internal architecture visibility
        'forward': forward,
        'backward': backward
    }

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

