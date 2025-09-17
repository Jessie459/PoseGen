import torch


# def otsu_threshold(images, num_bins=256):
#     """
#     Compute Otsu's threshold for grayscale images using PyTorch.

#     Args:
#         images: Tensor of shape (batch_size, height, width) with values in [0.0, 1.0]
#         num_bins: Number of histogram bins (default: 256)

#     Returns:
#         thresholds: Tensor of shape (batch_size,) containing optimal thresholds
#         binary_images: Tensor of shape (batch_size, height, width) with binary values
#     """
#     batch_size, height, width = images.shape
#     device = images.device

#     # Convert to integer bins for histogram computation
#     images_int = (images * (num_bins - 1)).long()

#     # Compute histograms for each image in the batch
#     histograms = torch.zeros(batch_size, num_bins, device=device)

#     for i in range(batch_size):
#         hist = torch.bincount(images_int[i].flatten(), minlength=num_bins)
#         histograms[i] = hist.float()

#     # Normalize histograms to get probabilities
#     total_pixels = height * width
#     probabilities = histograms / total_pixels

#     # Compute cumulative probabilities and means
#     cumulative_probs = torch.cumsum(probabilities, dim=1)

#     # Intensity levels (0 to num_bins-1)
#     intensity_levels = torch.arange(num_bins, device=device).float()

#     # Compute cumulative means
#     weighted_intensities = probabilities * intensity_levels.unsqueeze(0)
#     cumulative_means = torch.cumsum(weighted_intensities, dim=1)

#     # Total mean for each image
#     total_means = cumulative_means[:, -1:]

#     # Compute between-class variance for all possible thresholds
#     # Avoid division by zero by adding small epsilon
#     eps = 1e-8

#     # Background class weights and means
#     w0 = cumulative_probs
#     mu0 = torch.where(w0 > eps, cumulative_means / (w0 + eps), torch.zeros_like(w0))

#     # Foreground class weights and means
#     w1 = 1.0 - w0
#     mu1 = torch.where(w1 > eps, (total_means - cumulative_means) / (w1 + eps), torch.zeros_like(w1))

#     # Between-class variance
#     between_class_variance = w0 * w1 * (mu0 - mu1) ** 2

#     # Find optimal thresholds (argmax of between-class variance)
#     optimal_indices = torch.argmax(between_class_variance, dim=1)

#     # Convert back to [0, 1] range
#     thresholds = optimal_indices.float() / (num_bins - 1)

#     # Apply thresholding to create binary images
#     binary_images = (images > thresholds.view(-1, 1, 1)).float()

#     return thresholds, binary_images


def otsu_threshold(images: torch.Tensor, num_bins=256, return_thresholds=False):
    """
    Args:
        images (torch.Tensor): Shape: [batch_size, height, width]. Range: [0.0, 1.0]
        num_bins (int): Number of histogram bins.
    """
    device = images.device
    images = images.float()

    # Flatten images for histogram computation
    batch_size, height, width = images.shape
    images_flat = images.view(batch_size, -1)

    # Create bin edges
    bin_edges = torch.linspace(0, 1, num_bins + 1, device=device)

    # Compute histograms using torch.histc
    histograms = torch.stack([torch.histc(img, bins=num_bins, min=0.0, max=1.0) for img in images_flat])

    # Normalize to probabilities
    probabilities = histograms / (height * width)

    # Intensity levels (bin centers)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute cumulative statistics
    cumulative_probs = torch.cumsum(probabilities, dim=1)
    weighted_intensities = probabilities * bin_centers.unsqueeze(0)
    cumulative_means = torch.cumsum(weighted_intensities, dim=1)

    # Total mean
    total_means = cumulative_means[:, -1:]

    # Between-class variance computation
    eps = 1e-8
    w0 = cumulative_probs
    w1 = 1.0 - w0

    mu0 = torch.where(w0 > eps, cumulative_means / (w0 + eps), torch.zeros_like(w0))
    mu1 = torch.where(w1 > eps, (total_means - cumulative_means) / (w1 + eps), torch.zeros_like(w1))

    between_class_variance = w0 * w1 * (mu0 - mu1) ** 2

    # Find optimal thresholds
    optimal_indices = torch.argmax(between_class_variance, dim=1)
    thresholds = bin_centers[optimal_indices]

    # Apply thresholding
    binary_images = (images > thresholds.view(-1, 1, 1)).to(torch.uint8)

    if return_thresholds:
        return thresholds, binary_images
    return binary_images
