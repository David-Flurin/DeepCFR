import torch


def weighted_mse_loss(predictions, targets, w):
    """
    Compute the weighted mean squared error loss.
    
    Args:
        predictions (torch.Tensor): The predicted values.
        targets (torch.Tensor): The target values.
        weights (torch.Tensor): The weights for each sample.
        
    Returns:
        torch.Tensor: The computed loss.
    """
    
    # Compute the weighted mean squared error
    loss = torch.mean(w * ((predictions - targets) ** 2))
    
    return loss