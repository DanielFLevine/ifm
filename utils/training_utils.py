import torch

def generate_paths(X_0, X_1, sigma=0.0001, time_points=16, straight_paths=False):
    # Convert lists to tensors
    X_0 = torch.tensor(X_0, dtype=torch.float32)
    X_1 = torch.tensor(X_1, dtype=torch.float32)
    
    # Dimensions
    dim = len(X_0)  # Here dim is 5000
    
    # Generate time points: from 0 to 1, including both endpoints, evenly spaced
    times = torch.linspace(0, 1, steps=time_points).view(time_points, 1)
    
    # Expand times and inputs to broadcast across dimensions
    times_expanded = times.expand(time_points, dim)
    
    # Linear interpolation: tX_1 + (1-t)X_0 = X_0 + t(X_1 - X_0)
    path_means = X_0 + times_expanded * (X_1 - X_0)
    
    # Initialize paths with means (ensures exact start at X_0 and end at X_1)
    paths = path_means.clone()
    
    # Gaussian noise: zero mean, sigma standard deviation, but not for the first and last time points
    if straight_paths:
        return paths
    
    if time_points > 2:
        noise = sigma * torch.randn(time_points-2, dim)
        
        # Determine where X_0 or X_1 is non-zero, for intermediate time points
        non_zero_mask = ((X_0 != 0) | (X_1 != 0))
        non_zero_mask_expanded = non_zero_mask.unsqueeze(0).expand(time_points-2, -1)
        
        # Apply noise only where non_zero_mask is True, and only to intermediate points
        paths[1:-1] = paths[1:-1].where(~non_zero_mask_expanded, paths[1:-1] + noise)

    return paths


def copy_weights(pretrained_model, custom_model, num_hidden_layers, num_attention_heads, hidden_size, intermediate_size):
    # Copy the weights for each relevant layer
    for layer_idx in range(num_hidden_layers):
        # Access the pretrained layer
        pretrained_layer = pretrained_model.gpt_neox.layers[layer_idx]
        custom_layer = custom_model.gpt_neox.layers[layer_idx]

        # Copy attention weights from query_key_value
        pretrained_qkv = pretrained_layer.attention.query_key_value.weight.data
        custom_qkv = custom_layer.attention.query_key_value.weight.data

        # Slice the pretrained weights to fit the custom model's dimensions
        custom_qkv.copy_(pretrained_qkv[:custom_qkv.shape[0], :custom_qkv.shape[1]])
        
        pretrained_qkv_bias = pretrained_layer.attention.query_key_value.bias.data
        custom_qkv_bias = custom_layer.attention.query_key_value.bias.data
        custom_qkv_bias.copy_(pretrained_qkv_bias[:custom_qkv_bias.shape[0]])

        # Copy attention dense weights
        custom_dense_weight = custom_layer.attention.dense.weight.data
        pretrained_dense_weight = pretrained_layer.attention.dense.weight.data
        custom_dense_weight.copy_(pretrained_dense_weight[:custom_dense_weight.shape[0], :custom_dense_weight.shape[1]])
        
        custom_layer.attention.dense.bias.data.copy_(pretrained_layer.attention.dense.bias.data[:hidden_size])

        # Copy dense weights
        custom_layer.mlp.dense_h_to_4h.weight.data.copy_(pretrained_layer.mlp.dense_h_to_4h.weight.data[:intermediate_size, :hidden_size])
        custom_layer.mlp.dense_h_to_4h.bias.data.copy_(pretrained_layer.mlp.dense_h_to_4h.bias.data[:intermediate_size])
        
        custom_layer.mlp.dense_4h_to_h.weight.data.copy_(pretrained_layer.mlp.dense_4h_to_h.weight.data[:hidden_size, :intermediate_size])
        custom_layer.mlp.dense_4h_to_h.bias.data.copy_(pretrained_layer.mlp.dense_4h_to_h.bias.data[:hidden_size])

    # Copy the final layer norm weights
    custom_model.gpt_neox.final_layer_norm.weight.data.copy_(pretrained_model.gpt_neox.final_layer_norm.weight.data[:hidden_size])
    custom_model.gpt_neox.final_layer_norm.bias.data.copy_(pretrained_model.gpt_neox.final_layer_norm.bias.data[:hidden_size])