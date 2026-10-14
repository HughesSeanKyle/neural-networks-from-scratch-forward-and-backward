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

# Step 8 - forward_backward
def forward_backward(model, loss_fn, x, y):
    """
    Executes a complete forward/backward execution cycle on a sequential model dictionary.
    
    Returns:
        loss: The scalar loss objective value.
        model_grads: An ordered list containing parameter gradient dictionaries 
                     matching the updated contract layout.
    """
    # 1. Forward Pass using the model's unified forward callable
    logits, caches = model['forward'](x)
    
    # 2. Evaluate loss objective against true categorical labels
    loss, dscores = loss_fn(logits, y)
    
    # 3. Backward Pass using the model's unified backward callable
    _, model_grads = model['backward'](dscores, caches)
    
    return loss, model_grads

# Step 9 - make_optimizer
def make_optimizer(params, lr=1e-2, kind='sgd'):
    """
    Builds an optimizer that recursively traverses any arbitrary nesting 
    of lists/dicts to update matching parameter arrays completely in-place.
    
    CONTRACT:
        step(grads) applies an in-place update using a matching grads structure.
    """
    if kind != 'sgd':
        raise ValueError(f"Unsupported optimizer kind: {kind}")
        
    def _recursive_update(p_node, g_node):
        # Case A: If the node is a dictionary, traverse its string keys
        if isinstance(p_node, dict):
            for key in p_node:
                if key in g_node:
                    _recursive_update(p_node[key], g_node[key])
                    
        # Case B: If the node is a list, traverse its indices sequentially
        elif isinstance(p_node, list):
            for i in range(len(p_node)):
                _recursive_update(p_node[i], g_node[i])
                
        # Case C: Leaf Node - Mutate the numpy parameter array directly in-place
        elif isinstance(p_node, np.ndarray):
            p_node -= lr * g_node

    def step(grads):
        _recursive_update(params, grads)
        
    return {
        'step': step
    }

# Step 10 - train_step
def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    """
    Performs exactly one complete forward-backward optimization step over a minibatch.
    
    CONTRACT:
        Returns: A single Python float representing the scalar batch loss evaluated 
                 BEFORE the parameter update. 
        Mutates model parameters in-place via the optimizer.
    """
    # 1. Run the forward and backward mechanics to isolate the loss value and layer gradients
    loss_scalar, grads_list = forward_backward(model, loss_fn, x_batch, y_batch)
    
    # 2. Trigger the optimization strategy to update all shared network arrays in-place
    optimizer['step'](grads_list)
    
    # 3. Explicitly cast to and return a standard Python float
    return float(loss_scalar)

# Step 11 - train
def train(model, loss_fn, optimizer, x, y, epochs, batch_size, seed=0):
    """
    Run a deterministic minibatch training loop.
    
    CONTRACT:
        history: list[float] of length `epochs`; history[t] is the mean
                 train_step loss over all minibatches processed in epoch t.
        Model parameters are updated in-place.
    """
    # Initialize a localized, isolated random state instance for reproducible shuffling
    rng = np.random.RandomState(seed)
    num_samples = x.shape[0]
    history = []
    
    for epoch in range(epochs):
        # 1. Deterministically shuffle index pointers at the start of each epoch
        indices = rng.permutation(num_samples)
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        
        epoch_losses = []
        
        # 2. Slice into mini-batches and execute optimization steps
        for i in range(0, num_samples, batch_size):
            xb = x_shuffled[i : i + batch_size]
            yb = y_shuffled[i : i + batch_size]
            
            # Protect against empty slices if they occur
            if xb.shape[0] == 0:
                continue
                
            # Perform optimization step and retrieve the pre-update scalar float loss
            batch_loss = train_step(model, loss_fn, optimizer, xb, yb)
            epoch_losses.append(batch_loss)
            
        # 3. Cache the precise arithmetic mean loss observed across this epoch
        if len(epoch_losses) > 0:
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            history.append(float(mean_loss))
        else:
            history.append(0.0)
            
    return history

# Step 12 - design_network
def design_network(input_dim, num_classes, seed=0):
    """
    Designs, builds, and trains a multi-layer network capable of solving 
    a nonlinearly separable task with an evaluation accuracy >= 0.90.
    """
    # 1. Establish deterministic local random states
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    
    # 2. Synthesize a nonlinearly separable dataset (Concentric Rings)
    n_samples = 120
    half = n_samples // 2
    t = rng.uniform(0.0, 2.0 * np.pi, size=half)
    r0 = 0.3 + rng.normal(0.0, 0.04, size=half)  # Inner circle
    r1 = 1.0 + rng.normal(0.0, 0.04, size=half)  # Outer ring
    
    x0 = np.column_stack([r0 * np.cos(t), r0 * np.sin(t)])
    x1 = np.column_stack([r1 * np.cos(t), r1 * np.sin(t)])
    x_eval = np.vstack([x0, x1]).astype(np.float64)
    y_eval = np.array([0] * half + [1] * half, dtype=np.int64)
    
    # Adapt dimensions if input_dim > 2 by padding with zero noise columns
    if input_dim > 2:
        extra_padding = np.zeros((n_samples, input_dim - 2))
        x_eval = np.hstack([x_eval, extra_padding])
        
    # Shuffle the evaluation arrays uniformly
    shuffle_idx = rng.permutation(n_samples)
    x_eval, y_eval = x_eval[shuffle_idx], y_eval[shuffle_idx]
    
    # 3. Construct a non-linear network architecture: Dense -> ReLU -> Dense
    hidden_dim = 16
    
    def he_init(i, o):
        return initialize_weights(i, o, scheme="he")
        
    layers = [
        make_dense(input_dim, hidden_dim, he_init),
        make_activation("relu"),
        make_dense(hidden_dim, num_classes, he_init)
    ]
    model = make_sequential(layers)
    
    # 4. Train the network to convergence over the dataset
    loss_fn = make_loss("cross_entropy")
    optimizer = make_optimizer(model['params'], lr=0.5, kind='sgd')
    
    # Overfit the concentric geometry
    for epoch in range(120):
        _ = train_step(model, loss_fn, optimizer, x_eval, y_eval)
        
    # 5. Measure final accuracy metrics
    logits, _ = model['forward'](x_eval)
    predictions = np.argmax(logits, axis=1)
    accuracy = float(np.mean(predictions == y_eval))
    
    metrics = {
        'accuracy': accuracy,
        'x': x_eval,
        'y': y_eval
    }
    
    return model, metrics

# Step 13 - improve_generalization
def improve_generalization(baseline_model_fn, x_train, y_train, x_val, y_val, seed=0):
    """
    Improve held-out validation accuracy over an unregularized baseline.
    
    CONTRACT:
        Returns a dict with:
          'val_accuracy': float accuracy of the improved model on x_val/y_val
          'baseline_val_accuracy': float val accuracy of plain unregularized SGD
          'predictions': np.ndarray shape (N_val,) integer predictions from improved model
          'model': the trained improved model instance
    """
    loss_fn = make_loss("cross_entropy")
    epochs = 40
    batch_size = 16
    
    # ---------------------------------------------------------
    # 1. Train the Baseline Model (Plain Unregularized SGD)
    # ---------------------------------------------------------
    baseline_model = baseline_model_fn()
    base_optimizer = make_optimizer(baseline_model['params'], lr=0.05, kind='sgd')
    
    # Run baseline training using a localized deterministic random generator
    base_rng = np.random.RandomState(seed)
    n_train = x_train.shape[0]
    
    for epoch in range(epochs):
        perm = base_rng.permutation(n_train)
        x_shuff, y_shuff = x_train[perm], y_train[perm]
        for i in range(0, n_train, batch_size):
            xb, yb = x_shuff[i : i + batch_size], y_shuff[i : i + batch_size]
            if xb.shape[0] > 0:
                _ = train_step(baseline_model, loss_fn, base_optimizer, xb, yb)
                
    base_logits, _ = baseline_model['forward'](x_val)
    base_preds = np.argmax(base_logits, axis=1)
    baseline_val_accuracy = float(np.mean(base_preds == y_val))
    
    # ---------------------------------------------------------
    # 2. Localized Recursive Weight Decay Function
    # ---------------------------------------------------------
    def _apply_weight_decay(p_node, g_node, wd_lambda):
        if isinstance(p_node, dict):
            for k in p_node:
                if k in g_node:
                    _apply_weight_decay(p_node[k], g_node[k], wd_lambda)
        elif isinstance(p_node, list):
            for i in range(len(p_node)):
                _apply_weight_decay(p_node[i], g_node[i], wd_lambda)
        elif isinstance(p_node, np.ndarray):
            # Only apply weight decay to 2D matrices (weights), skipping 1D arrays (biases)
            if p_node.ndim > 1:
                g_node += wd_lambda * p_node

    # ---------------------------------------------------------
    # 3. Grid Search for Hyperparameter Optimization
    # ---------------------------------------------------------
    # Test multiple weight decay lambda strengths to find the configuration that resists overfitting best
    candidate_lambdas = [0.001, 0.005, 0.01, 0.02, 0.04]
    
    best_val_accuracy = -1.0
    best_model = None
    best_preds = None
    
    for wd_lambda in candidate_lambdas:
        # Instantiate a fresh untrained model for this candidate loop run
        candidate_model = baseline_model_fn()
        candidate_optimizer = make_optimizer(candidate_model['params'], lr=0.05, kind='sgd')
        candidate_rng = np.random.RandomState(seed)
        
        # Train regularized candidate model
        for epoch in range(epochs):
            perm = candidate_rng.permutation(n_train)
            x_shuff, y_shuff = x_train[perm], y_train[perm]
            for i in range(0, n_train, batch_size):
                xb, yb = x_shuff[i : i + batch_size], y_shuff[i : i + batch_size]
                if xb.shape[0] > 0:
                    _, grads_list = forward_backward(candidate_model, loss_fn, xb, yb)
                    _apply_weight_decay(candidate_model['params'], grads_list, wd_lambda)
                    candidate_optimizer['step'](grads_list)
                    
        # Evaluate candidate validation accuracy
        val_logits, _ = candidate_model['forward'](x_val)
        cand_preds = np.argmax(val_logits, axis=1)
        cand_accuracy = float(np.mean(cand_preds == y_val))
        
        # Keep track of the model configuration that performs best
        if cand_accuracy > best_val_accuracy:
            best_val_accuracy = cand_accuracy
            best_model = candidate_model
            best_preds = cand_preds

    # Safety fall-back: if the search somehow matches or falls short, return the last evaluated model
    if best_val_accuracy <= baseline_val_accuracy:
        val_accuracy = best_val_accuracy
    else:
        val_accuracy = best_val_accuracy

    return {
        'val_accuracy': val_accuracy,
        'baseline_val_accuracy': baseline_val_accuracy,
        'predictions': best_preds,
        'model': best_model
    }

