import numpy as np

def add_noise_by_snr_numpy(data: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add Gaussian noise to a multi-dimensional time series based on SNR
    
    Args:
        data: numpy array with shape (T, N), where T is the time length and N is the feature dimension
        snr_db: target signal-to-noise ratio (dB)
    
    Returns:
        Data with added noise
    """
    # Calculate the signal power for each feature dimension
    signal_power = np.mean(data**2, axis=0)  # shape: (N,)
    
    # Calculate the required noise power based on SNR
    noise_power = signal_power / (10 ** (snr_db / 10))  # shape: (N,)
    
    # Generate Gaussian noise
    noise = np.random.normal(0, 1, size=data.shape)  # shape: (T, N)
    
    # Scale the noise amplitude
    scaled_noise = noise * np.sqrt(noise_power)  # shape: (T, N)
    
    # Add noise
    noisy_data = data + scaled_noise

    # print("The amplitude of noise is: ", np.sqrt(noise_power))
    
    return noisy_data

def verify_snr_numpy(clean_data: np.ndarray, noisy_data: np.ndarray) -> np.ndarray:
    """
    Verify the actual SNR for each feature dimension
    
    Returns:
        Actual SNR (dB) for each feature dimension
    """
    noise = noisy_data - clean_data
    signal_power = np.mean(clean_data**2, axis=0)
    noise_power = np.mean(noise**2, axis=0)
    actual_snr = 10 * np.log10(signal_power / noise_power)
    return actual_snr
