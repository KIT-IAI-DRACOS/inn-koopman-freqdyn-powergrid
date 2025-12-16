import torch
import warnings
import numpy as np

class StableKoopmanOperator(torch.nn.Module):
    """
    Stable Koopman operator, ensures all eigenvalues are stable via special parameterization.
    
    Uses matrix decomposition to ensure all eigenvalue real parts are negative:
    - K = K1 + K2, where:
      - K1 is a diagonal matrix with diagonal entries -a^2 (non-positive)
      - K2 is a skew-symmetric matrix (K2 = B - B^T), where B is an upper-triangular matrix
    
    This parameterization ensures all eigenvalues of K have non-positive real parts, while providing strong expressiveness.
    
    Parameters
    ----------
    dim : int
        Dimension of the Koopman operator
    dt : float, default=1.0
        Time step
    bandwidth : int, default=None
        Controls the structural complexity of the matrix:
        - 1: diagonal elements only (pure diagonal matrix)
        - 2: diagonal and first off-diagonal
        - 3: diagonal, first and second off-diagonals
        - None or <=0 or >=dim: no restriction (full matrix)    
    """
    def __init__(self,
        dim: int,
        dt: float = 1.0,
        bandwidth: int = None):

        super().__init__()
        self.dim = dim
        self.register_buffer("dt", torch.tensor(dt))
        self.bandwidth = bandwidth
        
        # Initialize parameters for diagonal matrix K1 (a parameters)
        # Random initialization, ensures diagonal entries are -a^2 (non-positive)
        r_scale = 3  # Decay times within r_scale*dt will be ignored
        
        # Generate random radius (using sqrt(r) for uniform distribution)
        r = torch.pow(torch.sqrt(torch.rand(self.dim)), 1 / (r_scale * self.dt))
        
        # Ensure r is not too small (avoid numerical instability)
        zeros_indices = torch.nonzero(r < 1e-5).view(-1)
        if zeros_indices.any():
            r[zeros_indices] = 1e-5
        log_r = torch.log(r)
        
        # Compute a parameters
        self.register_parameter(
            "a_params",
            torch.nn.Parameter(
                torch.sqrt(-log_r)
            ),
        )
        
        # Initialize upper-triangular part of skew-symmetric matrix K2 (B parameters)
        # Only store upper-triangular part since K2 = B - B^T
        # Number of nonzero upper-triangle elements is dim*(dim-1)/2
        n_upper_elements = (self.dim * (self.dim - 1)) // 2

        # Use small initial values to avoid large skew-symmetric part at start
        self.register_parameter(
            "b_params",
            torch.nn.Parameter(
                torch.randn(n_upper_elements) * 0.5
            ),
        )

    def get_K(self):
        """Construct Koopman operator matrix K = K1 + K2"""
        # Initialize zero matrix
        K1 = torch.zeros(self.dim, self.dim, device=self.a_params.device)
        K2 = torch.zeros_like(K1)
        
        # Construct diagonal matrix K1
        K1.diagonal().copy_(-torch.pow(self.a_params, 2))

        # Construct skew-symmetric matrix K2
        # Check bandwidth validity
        if self.bandwidth is None or self.bandwidth <= 0 or self.bandwidth >= self.dim:
            # Use full skew-symmetric matrix (original implementation)
            triu_indices = torch.triu_indices(self.dim, self.dim, offset=1)
            K2[triu_indices[0], triu_indices[1]] = self.b_params
        else:
            # Banded skew-symmetric matrix implementation
            # Create a mask to determine which elements to fill
            # bandwidth=1: diagonal only (no skew-symmetric part)
            # bandwidth=2: diagonal and first off-diagonal
            # bandwidth=3: diagonal, first and second off-diagonals, etc.
            
            param_idx = 0
            for offset in range(1, min(self.bandwidth, self.dim)):
                # Number of elements on this diagonal
                diag_size = self.dim - offset

                # Get indices for this diagonal
                i_indices = torch.arange(0, diag_size, device=self.a_params.device)
                j_indices = i_indices + offset
                
                # Fill corresponding positions in K2
                if param_idx + diag_size <= len(self.b_params):
                    K2[i_indices, j_indices] = self.b_params[param_idx:param_idx + diag_size]
                    param_idx += diag_size

        K2 = K2 - K2.transpose(0, 1)  # Diagonal remains zero
        
        # Combine K1 and K2
        K = K1 + K2
        return K

    def get_discrete_time_Koopman_Operator(self):
        """Compute discrete-time Koopman operator"""
        if not hasattr(self, "dt") or self.dt is None:
            return self.get_K()
        
        # # Get continuous-time Koopman operator
        # K = self.get_K()
        # # Compute matrix exponential e^(K*dt)
        # discrete_K = torch.matrix_exp(K * self.dt)

        eigenvalues, eigenvectors, eigenvectors_inv = self.get_eigensystems()
        discrete_eigen = torch.exp(eigenvalues * self.dt)
        discrete_K = eigenvectors @ torch.diag(discrete_eigen) @ eigenvectors_inv

        return discrete_K, discrete_eigen, eigenvectors, eigenvectors_inv

    def get_eigensystems(self):
        """Compute eigenvalues and eigenvectors of the Koopman operator"""
        # Get Koopman matrix
        K = self.get_K()
        
        # Compute eigenvalues and eigenvectors
        # Note: torch.linalg.eig returns complex eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eig(K)

        
        # Compute inverse eigenvector matrix
        eigenvectors_inv = torch.linalg.inv(eigenvectors)


        
        return eigenvalues, eigenvectors, eigenvectors_inv
    
    


    def forward(self, encoded_x, n = 1, record = False):
        """Apply Koopman operator for forward propagation
        phi: shape [batch_size, seq_len, n_Koopman] or [seq_len, n_Koopman]"""

        if n == 0:
            return encoded_x
        
        #eigenvalues, eigenvectors, eigenvectors_inv = self.get_eigensystems()
        # self.get_K() == eigenvectors @ torch.diag(eigenvalues) @ eigenvectors_inv
        #print('')
        #print('self.get_K() error: ' ,(abs(self.get_K() - eigenvectors @ torch.diag(eigenvalues) @ eigenvectors_inv).max()))
        K_T = self.get_K()
        if not hasattr(self, "dt") or self.dt is None:
            K_T_n_real = self.get_K().T
        else:
            if not record:
                #eigenvalues_n = torch.exp(eigenvalues * self.dt * n)
                #K_T_n = eigenvectors_inv.T @ torch.diag(eigenvalues_n) @ eigenvectors.T

                K_T_n = torch.linalg.matrix_exp(K_T * self.dt * n).T
                #print('K_T_n error 1',(abs(K_T_n_2-K_T_n)).max())
                # Handle complex values (result should be real for our Koopman structure)
                K_T_n_real = K_T_n.real
                if torch.is_complex(K_T_n):
                    # Check if imaginary part is negligible
                    imag_ratio = torch.max(torch.abs(K_T_n.imag)) / (torch.max(torch.abs(K_T_n.real)) + 1e-10)
                    if imag_ratio > 1e-6:
                        warnings.warn(f"Large imaginary component in K_T_n: {imag_ratio}")
            else:
                K_T = torch.linalg.matrix_exp(K_T * self.dt).T
                K_T_n = torch.zeros(n, self.dim, self.dim, dtype=torch.complex64, device=self.a_params.device)
                #K_T_n_2 = torch.zeros(n, self.dim, self.dim, dtype=torch.complex64, device=self.a_params.device)
                for i in range(n):
                    #eigenvalues_n = torch.exp(eigenvalues * self.dt * (i+1))
                    #K_T_n[i] = eigenvectors_inv.T @ torch.diag(eigenvalues_n) @ eigenvectors.T
                    # K_T_n[i] = torch.linalg.matrix_exp(K_T*(i+1)* self.dt).T
                    K_T_n[i] = torch.linalg.matrix_power(K_T, i+1)
                    #print('K_T_n error 2', (abs(K_T_n_2-K_T_n)).max())
                    if torch.is_complex(K_T_n):
                        # Check if imaginary part is negligible
                        imag_ratio = torch.max(torch.abs(K_T_n[i].imag)) / (torch.max(torch.abs(K_T_n[i].real)) + 1e-10)
                        if imag_ratio > 1e-6:
                            warnings.warn(f"Large imaginary component in K_T_n: {imag_ratio}")
                K_T_n_real = K_T_n.real

        if encoded_x.dim() == 2 and not record:                         # shape of encoded_x: [others, n_Koopman]
            result = torch.matmul(encoded_x, K_T_n_real)
        elif encoded_x.dim() == 3 and not record:                       # shape of encoded_x: [batch_size, seq_len, n_Koopman]
            result = torch.einsum("bik,kl->bil", encoded_x, K_T_n_real) 
        elif encoded_x.dim() == 3 and record:                           # shape of encoded_x: [batch_size, seq_len, n_Koopman]
            result = torch.einsum("bk,ikl->bil", encoded_x[:,-1,:], K_T_n_real)
        else:
            raise ValueError(f"Unsupported input shape: {encoded_x.shape}")
        return result

    def inverse(self, encoded_x, n = 1, record = False):
        """Apply Koopman operator for backward propagation
        phi: shape [batch_size, seq_len, n_Koopman] or [seq_len, n_Koopman]"""

        if n == 0:
            return encoded_x
        
        eigenvalues, eigenvectors, eigenvectors_inv = self.get_eigensystems()
        # self.get_K() == eigenvectors @ torch.diag(eigenvalues) @ eigenvectors_inv

        if not hasattr(self, "dt") or self.dt is None:
            K_T_n_real = eigenvectors @ torch.diag(-eigenvalues) @ eigenvectors_inv
            K_T_n_real = K_T_n_real.T
        else:
            if not record:
                eigenvalues_n = torch.exp(-eigenvalues * self.dt * n)
                K_T_n = eigenvectors_inv.T @ torch.diag(eigenvalues_n) @ eigenvectors.T
                # Handle complex values (result should be real for our Koopman structure)
                K_T_n_real = K_T_n.real
                if torch.is_complex(K_T_n):
                    # Check if imaginary part is negligible
                    imag_ratio = torch.max(torch.abs(K_T_n.imag)) / (torch.max(torch.abs(K_T_n.real)) + 1e-10)
                    if imag_ratio > 1e-6:
                        warnings.warn(f"Large imaginary component in K_T_n: {imag_ratio}")
            else:
                K_T_n = torch.zeros(n, self.dim, self.dim, dtype=torch.complex64, device=self.a_params.device)
                for i in range(n):
                    eigenvalues_n = torch.exp(-eigenvalues * self.dt * (i+1))
                    K_T_n[i] = eigenvectors_inv.T @ torch.diag(eigenvalues_n) @ eigenvectors.T
                    if torch.is_complex(K_T_n):
                        # Check if imaginary part is negligible
                        imag_ratio = torch.max(torch.abs(K_T_n[i].imag)) / (torch.max(torch.abs(K_T_n[i].real)) + 1e-10)
                        if imag_ratio > 1e-6:
                            warnings.warn(f"Large imaginary component in K_T_n: {imag_ratio}")
                K_T_n_real = K_T_n.real

        if encoded_x.dim() == 2 and not record:                         # shape of encoded_x: [others, n_Koopman]
            result = torch.matmul(encoded_x, K_T_n_real)
        elif encoded_x.dim() == 3 and not record:                       # shape of encoded_x: [batch_size, seq_len, n_Koopman]
            result = torch.einsum("bik,kl->bil", encoded_x, K_T_n_real) 
        elif encoded_x.dim() == 3 and record:                           # shape of encoded_x: [batch_size, seq_len, n_Koopman]
            result = torch.einsum("bk,ikl->bil", encoded_x[:,-1,:], K_T_n_real)
        else:
            raise ValueError(f"Unsupported input shape: {encoded_x.shape}")
        return result

    def _initialize_koopman_from_eigenvalues(self,
        eigenvalues: np.ndarray,
        dt: float,
        device: str = "cpu"
    ):
        """
        Initialize StableKoopmanOperator parameters from HODMD eigenvalues
        
        Parameters
        ----------
        self : StableKoopmanOperator
            Koopman operator to initialize
        eigenvalues : np.ndarray
            Eigenvalues computed by HODMD
        dt : float
            Time step
        device : str
            Computation device
        """
        # Ensure number of eigenvalues does not exceed Koopman dimension
        n_eigs = min(len(eigenvalues), self.dim)
        
        # Compute continuous-time eigenvalues (from discrete-time eigenvalues)
        # Discrete-time eigenvalue λ_d = exp(λ_c * dt)
        # Continuous-time eigenvalue λ_c = log(λ_d) / dt
        cont_eigenvalues = np.log(eigenvalues[:n_eigs]) / dt
        
        # Extract real and imaginary parts
        real_parts = np.real(cont_eigenvalues)
        imag_parts = np.imag(cont_eigenvalues)
        
        # Ensure all eigenvalue real parts are negative (for stability)
        real_parts = np.minimum(real_parts, -1e-4)
        
        # Initialize diagonal parameter (a_params)
        # Diagonal entry is -a^2, corresponding to the real part of eigenvalue
        a_params = torch.tensor(np.sqrt(-real_parts), device=device)
        
        # Update a_params
        self.a_params.data = a_params
        
        # Note: the skew-symmetric part (b_params) cannot be directly inferred from eigenvalues
        # We can keep the random initialization or set them to small values
        # Here, we choose to set b_params to small random values
        self.b_params.data = torch.randn_like(self.b_params) * 0.01
        
        # If imaginary parts are available, they can be used to initialize b_params
        # But this requires more complex computation, which is omitted here
        
        print(f"Koopman operator parameters initialized: a_params={self.a_params[:5]}...")
        print(f"b_params kept as small random values, shape={self.b_params.shape}")

class OrthogonalLinear(torch.nn.Module):
    # Householder Orthogonal Linear with block swapping capability
    def __init__(self, dim, init_mode='orthogonal', n_reflections = None, bias=True, block1_size=None):
        super(OrthogonalLinear, self).__init__()
        self.dim = dim
        self.init_mode = init_mode
        
        # Set default number of reflections
        if n_reflections is None:
            n_reflections = min(dim, 30)
        
        # Set default block1_size
        if block1_size is None and init_mode == 'swap_blocks':
            raise ValueError("block1_size must be specified when init_mode is 'swap_blocks'")
        else:
            self.block1_size = block1_size
        
        # Initialize reflection vectors
        if init_mode == 'swap_blocks' and self.block1_size is not None:
            # Validate parameters
            if self.block1_size >= dim:
                raise ValueError(f"block1_size ({self.block1_size}) must be less than dim ({dim})")
            
            # Create initial permutation matrix
            perm_matrix = self._create_swap_blocks_matrix()
            
            # Use polar decomposition to initialize Householder vectors
            # This will make the result of Householder transforms close to the desired permutation matrix
            self.vectors = torch.nn.Parameter(self._initialize_vectors_for_matrix(perm_matrix, n_reflections))
            
        elif init_mode == 'swap_blocks':
            # Original random initialization
            self.vectors = torch.nn.Parameter(torch.randn(n_reflections, dim))
        elif init_mode == 'orthogonal':
            # Parameterize Householder vectors
            self.vectors = torch.nn.Parameter(torch.randn(n_reflections, dim))
            self.bias = torch.nn.Parameter(torch.zeros(dim)) if bias else None
        else:    
            raise ValueError(f"Unknown init_mode: {init_mode}. Use 'orthogonal' or 'swap_blocks'")
        
        self.bias = torch.nn.Parameter(torch.zeros(dim)) if bias else None
    
    def _create_swap_blocks_matrix(self):
        """Create block swapping permutation matrix"""
        block1_size = self.block1_size
        # block2_size = self.dim - block1_size
        
        # Construct permutation indices
        indices = torch.cat([
            torch.arange(block1_size, self.dim),  # Move the later block to the front
            torch.arange(block1_size)             # Move the front block to the back
        ])
        
        # Create permutation matrix
        perm_matrix = torch.zeros(self.dim, self.dim)
        perm_matrix[torch.arange(self.dim), indices] = 1.0
        
        return perm_matrix
    
    def _initialize_vectors_for_matrix(self, target_matrix, n_reflections):
        """Initialize Householder vectors so that their product approximates the target matrix"""
        # Note: This is an approximate method. We use a series of Householder reflections
        # to iteratively approach the target matrix
        
        # Start from identity matrix
        current = torch.eye(self.dim)
        target = target_matrix
        
        # Compute the difference between target and current
        diff = target - current
        
        # Create Householder vectors
        vectors = torch.zeros(n_reflections, self.dim)
        
        for i in range(min(n_reflections, self.dim)):
            # Find the row with the largest norm in the difference matrix
            row_norms = torch.norm(diff, dim=1)
            max_row = torch.argmax(row_norms)
            
            # Use this row as the direction of the Householder vector
            v = diff[max_row].clone()
            
            # Normalize
            v_norm = torch.norm(v)
            if v_norm > 1e-6:  # Avoid division by zero
                v = v / v_norm
                
                # Add some random noise for diversity
                v = v + torch.randn_like(v) * 0.01
                v = v / torch.norm(v)
                
                # Update current matrix
                H = torch.eye(self.dim) - 2 * torch.outer(v, v)
                current = current @ H
                diff = target - current
            
            vectors[i] = v
        
        return vectors
    
    def _get_orthogonal_matrix(self):
        # Initialize as identity matrix
        Q = torch.eye(self.dim, device=self.vectors.device)
        
        # Apply each Householder transform
        for v in self.vectors:
            v_normalized = v / (torch.norm(v) + 1e-8)
            # Householder matrix: H = I - 2 * vv^T
            H = torch.eye(self.dim, device=v.device) - 2 * torch.outer(v_normalized, v_normalized)
            Q = Q @ H
        
        return Q
    
    def forward(self, x):
        weight = self._get_orthogonal_matrix()
        output = x @ weight
        if self.bias is not None:
            output = output + self.bias
        return output
    
    def inverse(self, y):
        weight = self._get_orthogonal_matrix()
        if self.bias is not None:
            y = y - self.bias
        return y @ weight.t()

    def _initialize_orthogonal_from_modes(self,
        modes: np.ndarray,
        input_dim: int,
        koopman_dim: int,
        device: str = "cpu"
    ):
        """
        Initialize OrthogonalLinear parameters from HODMD modes
        
        Parameters
        ----------
        self : OrthogonalLinear
            The orthogonal transform to initialize
        modes : np.ndarray
            Modes computed by HODMD
        input_dim : int
            Input feature dimension
        koopman_dim : int
            Koopman space dimension
        device : str
            Computation device
        """
        # Extract useful information from modes to initialize the orthogonal transform
        # Typically, we can use SVD to extract orthogonal bases
        
        # Ensure mode dimensions are appropriate
        n_modes = min(modes.shape[1], koopman_dim)
        
        # Perform SVD on the mode matrix
        try:
            U, _, _ = np.linalg.svd(modes[:input_dim, :n_modes], full_matrices=False)
            
            # Ensure U is an orthogonal matrix
            if U.shape[1] < koopman_dim:
                # If U does not have enough columns, pad it
                padding = np.random.randn(U.shape[0], koopman_dim - U.shape[1])
                # Orthogonalize new columns
                q, _ = np.linalg.qr(padding - U @ (U.T @ padding))
                U = np.hstack([U, q])
            
            # Convert to PyTorch tensor
            U_tensor = torch.tensor(U, dtype=torch.float32, device=device)
            
            # Use Householder transforms to approximate this orthogonal matrix
            # Here, we approximate the target orthogonal matrix by setting vectors
            target_matrix = U_tensor.t() @ U_tensor  # Should be close to identity

            # Initialize Householder vectors
            for i in range(min(self.vectors.shape[0], U.shape[1])):
                v = torch.tensor(U[:, i], dtype=torch.float32, device=device)
                # Add random noise to avoid singularity
                v = v + torch.randn_like(v) * 0.01
                # Normalize
                v = v / torch.norm(v)
                self.vectors.data[i] = v
                
            print(f"Orthogonal transform initialized from modes, used {n_modes} modes")
            
        except Exception as e:
            print(f"Error initializing orthogonal transform from modes: {str(e)}")
            print("Using random initialization as fallback")
            # Use random initialization as fallback
            torch.nn.init.orthogonal_(self.vectors)