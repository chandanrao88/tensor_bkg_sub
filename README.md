# Background Subtraction using Adaptive Tensor SVD

**Owner:**
- Chandan Rao

## Abstract

This report presents our implementation of a background subtraction algorithm using adaptive tensor singular value decomposition (SVD). We compare two tensor-product approaches: the T-product and M-product methods for efficient online updating of the background model. Our implementation processes video frames to separate foreground objects from the background, with experimental results showing comparative performance metrics between the two methods.

## Introduction

Background subtraction is a fundamental technique in computer vision that aims to separate moving objects (foreground) from static scenes (background) in video sequences. Traditional approaches often use matrix-based methods, but tensor-based approaches offer advantages for multi-dimensional data processing.

In this project, an adaptive tensor SVD algorithm has been implemened for background subtraction that:
* Represents video frames as third-order tensors
* Efficiently updates the background model using T-product and M-product methods
* Adapts to gradually changing scenes by incrementally updating the tensor SVD
* Compares performance metrics between T-product and M-product approaches

## Intuition Behind the Algorithm

The core intuition behind background subtraction using adaptive SVD builds on a fundamental observation: in video sequences captured by a static camera, the background remains relatively constant over time while foreground objects (people, vehicles, etc.) appear as sparse, moving elements. This characteristic allows us to model the problem mathematically:

* The static background naturally has a low-rank structure, as it consists of highly correlated pixel values across frames
* Moving foreground objects represent sparse deviations from this low-rank model
* Gradually changing background elements (like lighting variations) can be captured through adaptive updating of the low-rank model

This insight leads to formulating background subtraction as a mathematical decomposition problem where we separate a video sequence into a low-rank component (background) and a sparse component (foreground). Singular Value Decomposition (SVD) is particularly well-suited for this task as it provides a way to extract the principal components of the data while filtering out noise and sparse anomalies.

By extending this concept to tensors, we can better preserve the spatial-temporal structure of video data that would otherwise be lost when flattening frames into matrices. The tensor-based approach allows us to capture correlations across both spatial dimensions and time simultaneously, providing a more comprehensive model of the background.

## Methodology

### Mathematical Formulation

The foundation of our approach lies in the matrix decomposition problem:

A = L + S

where A represents the video data, L is the low-rank background component, and S is the sparse foreground component. This can be formulated as an optimization problem:

min(L,S) rank(L) + λ‖S‖₀ s.t. A = L + S

where λ is a regularization parameter and ‖S‖₀ represents the L0 norm (number of non-zero elements in S).

Since this problem is computationally infeasible, a relaxed formulation is used:

min ‖A−L‖F s.t. rank(L) ≤ k

where ‖.‖F denotes the Frobenius norm and k is a rank constraint.

Using Singular Value Decomposition (SVD), we can decompose the matrix A as:

A = UΣVᵀ

where U and V are orthogonal matrices, and Σ is a diagonal matrix of singular values.

In our tensor-based approach, these concepts can extend to third-order tensors using two different tensor product definitions.

### Tensor Representation

A video sequence as is represeted as a third-order tensor where:
* First two dimensions correspond to spatial coordinates (height × width)
* Third dimension corresponds to time (frames)
* For color videos, processed each channel separately

### T-product Based Tensor SVD

The T-product approach operates in the Fourier domain, defining the tensor SVD as:

A = U * S * Vᵀ

where * denotes the T-product, U and V are orthogonal tensors, and S is a f-diagonal tensor.

The process includes:
1. Transform the tensor to the Fourier domain along the third dimension: Â = fft(A,[],3)
2. Perform standard SVD on each frontal slice in the Fourier domain: Â(i) = Û(i)Ŝ(i)(V̂(i))ᵀ
3. Truncate components based on a threshold τstar applied to singular values
4. Transform back to the spatial domain: U = ifft(Û,[],3), S = ifft(Ŝ,[],3), V = ifft(V̂,[],3)

### M-product Based Tensor SVD

The M-product approach uses a transformation matrix M to define a different tensor product:

A = U ×ₘ S ×ₘ Vᵀ

where ×ₘ denotes the M-product.

The process includes:
1. Apply mode-3 product with matrix M to transform the tensor: Aₘ = A ×₃ M
2. Perform standard SVD on each frontal slice of the transformed tensor
3. Truncate components based on a threshold applied to singular values
4. Apply inverse mode-3 product to return to the original domain: A = Aₘ ×₃ M⁻¹

### Adaptive Background Model

The background model is adaptively updated following these key components:

1. **SVD Initialization (SVDComp)**: The initial set of frames is used to compute the SVD, capturing the dominant singular vectors that represent the background.
2. **Iterative Update (SVDAppend)**: New frames are appended in blocks of size β, with computational efficiency achieved through thresholding of singular values to determine the relevance of new information.
3. **Re-Initialization Strategies**: When the number of singular vectors exceeds a predefined limit nstar, re-initialization is performed either by using projected background images or directly through truncated singular vector sets.
4. **Normalization**: To avoid numerical instability and ensure consistent scaling, the singular values are normalized based on a predefined threshold τ.

The algorithm accommodates gradual scene changes by:

Bₜ = Uₜ * Sₜ * Vₜᵀ

where Bₜ is the background estimate at time t.

## Implementation

The algorithm has been implemented in Python using NumPy for tensor operations and OpenCV for image processing. The implementation consists of several key functions:

* `tsvd_comp` and `msvd_comp`: Compute T-product and M-product SVD respectively
* `tsvd_append` and `msvd_append`: Update SVD components with new data
* `T_projection` and `M_projection`: Project frames onto background subspace
* `TProd` and `MProd`: Compute T-product and M-product of tensors
* `process_image_sequence`: Main pipeline for background subtraction

Also created a validation implementation in MATLAB to verify our results and ensure correctness of both methods across different programming environments.

### Parameters

Key parameters in our implementation include:
* τstar: Controls the truncation of singular values
* τ: Threshold for SVD updating
* θ: Threshold for foreground detection
* β: Batch size for SVD updating
* nstar: Maximum rank before reinitialization

## Experimental Results

### Dataset

The "streetLight" image sequence from the CDnet 2014 (Change Detection) dataset for background subtraction. The test included:
* Initialization with 100 frames
* Processing up to 800 frames
* Color image processing with RGB channels

### Performance Metrics

Evaluated the performance using:
* Structural Similarity Index (SSIM) between original and background
* Peak Signal-to-Noise Ratio (PSNR) for background quality
* Jaccard similarity between foreground masks
* Computation time for T-product vs. M-product
* Model rank over time

### Comparison Results

Our experiments demonstrated:
* T-product method showed better foreground detection in scenarios with gradual background changes
* M-product method demonstrated faster computation time in most test cases
* The average model rank was typically lower with the T-product approach
* Both methods provided similar quality in terms of SSIM and PSNR metrics
* Results obtained from our Python implementation were consistent with our MATLAB validation, confirming the correctness of both approaches

## Conclusion

Our implementation of adaptive tensor SVD for background subtraction demonstrates the effectiveness of tensor-based approaches for video surveillance applications. The key findings include:

* Both T-product and M-product methods successfully capture the low-rank background structure while identifying sparse foreground objects
* The adaptive updating mechanism efficiently handles gradual changes in the background, such as illumination variations
* The tensor-based approach preserves spatial-temporal correlations that would be lost in matrix-based methods
* The computational efficiency achieved through strategic reinitialization and thresholding makes these methods viable for practical applications

The T-product approach generally provided slightly better quality in background estimation, likely due to its better preservation of the original tensor structure. Meanwhile, the M-product approach offered computational advantages, making it more suitable for applications with stricter time constraints.

Our implementation validates the core intuition that background subtraction can be effectively formulated as a low-rank plus sparse decomposition problem. By extending this concept to tensors, we capture richer spatial-temporal relationships in the video data.

# Applications of Background Subtraction Technology

## Security and Monitoring
- **Traffic Analysis**: Automated vehicle counting, congestion detection, and wrong-way driver identification
- **Security Surveillance**: Intelligent monitoring that adapts to environmental changes while detecting unauthorized presence

## Retail Intelligence
- **Customer Behavior Analysis**: Creating store traffic heat maps and tracking movement patterns
- **Queue Optimization**: Real-time staffing adjustments based on automated line length detection

## Industrial Applications
- **Production Monitoring**: Non-disruptive anomaly detection in manufacturing processes
- **Quality Assurance**: Automated inspection for missing or misaligned components

## Healthcare Solutions
- **Patient Safety**: Privacy-preserving monitoring for falls or concerning immobility
- **Rehabilitation**: Objective measurement of movement patterns and recovery progress

## Public Space Management
- **Crowd Dynamics**: Early detection of dangerous congestion or unusual crowd movements
- **Social Distancing**: Automated monitoring of interpersonal spacing in public areas

## Smart Environment Systems
- **Gesture Control**: Touchless interaction through movement detection
- **Occupancy-Based Automation**: Energy-efficient building management through presence detection

## Interactive Experiences
- **Responsive Installations**: Creating engaging exhibits that react to visitor movements
- **Simplified Motion Capture**: Basic movement tracking for games and interactive media without specialized equipment