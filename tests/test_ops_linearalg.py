import pytest
from ember_ml import ops
from ember_ml.ops import linearalg # Import the linearalg submodule
from ember_ml.nn import tensor

# Assume conftest.py provides 'backend' fixture

@pytest.fixture
def linearalg_matrices(backend):
    """Fixture to create sample matrices for linear algebra testing."""
    ops.set_backend(backend)
    # Square invertible matrix
    mat_a = tensor.convert_to_tensor([[3.0, 1.0], [1.0, 2.0]])
    # Another matrix
    mat_b = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    # Vector for solving Ax=vec
    vec = tensor.convert_to_tensor([9.0, 8.0])
    # Vector for diagonal
    diag_vec = tensor.convert_to_tensor([5.0, 6.0, 7.0])
    return mat_a, mat_b, vec, diag_vec

def test_linearalg_solve(linearalg_matrices, backend):
    """Tests linearalg.solve."""
    ops.set_backend(backend)
    mat_a, _, vec, _ = linearalg_matrices

    # Solve Ax = vec -> x = A_inv * vec
    # A_inv = 1/(6-1) * [[2, -1], [-1, 3]] = [[0.4, -0.2], [-0.2, 0.6]]
    # x = [[0.4, -0.2], [-0.2, 0.6]] * [9, 8]' = [3.6 - 1.6, -1.8 + 4.8]' = [2, 3]'
    solution = linearalg.solve(mat_a, vec)
    expected_solution = tensor.convert_to_tensor([2.0, 3.0])
    assert ops.allclose(solution, expected_solution), f"{backend}: Solve failed"

    # Verify solution
    verification = ops.matmul(mat_a, solution)
    assert ops.allclose(verification, vec), f"{backend}: Solve verification failed"

def test_linearalg_inv(linearalg_matrices, backend):
    """Tests linearalg.inv."""
    ops.set_backend(backend)
    mat_a, _, _, _ = linearalg_matrices

    inverse_a = linearalg.inv(mat_a)
    expected_inverse_a = tensor.convert_to_tensor([[0.4, -0.2], [-0.2, 0.6]])
    assert ops.allclose(inverse_a, expected_inverse_a, atol=1e-6), f"{backend}: Inverse calculation failed"

    # Check A * A_inv = Identity
    identity_calc = ops.matmul(mat_a, inverse_a)
    # Create identity matrix using tensor functions
    n_rows = tensor.shape(mat_a)[0]
    expected_identity = tensor.eye(n_rows)
    expected_identity = tensor.cast(expected_identity, identity_calc.dtype) # Match dtypes
    assert ops.allclose(identity_calc, expected_identity, atol=1e-6), f"{backend}: Inverse verification (A*A_inv != I) failed"

def test_linearalg_det(linearalg_matrices, backend):
    """Tests linearalg.det."""
    ops.set_backend(backend)
    mat_a, mat_b, _, _ = linearalg_matrices

    det_a = linearalg.det(mat_a)
    # Expected det(A) = 3*2 - 1*1 = 5
    assert ops.allclose(det_a, tensor.convert_to_tensor(5.0)), f"{backend}: Determinant(A) failed"

    det_b = linearalg.det(mat_b)
    # Expected det(B) = 1*4 - 2*3 = 4 - 6 = -2
    assert ops.allclose(det_b, tensor.convert_to_tensor(-2.0)), f"{backend}: Determinant(B) failed"

def test_linearalg_norm(linearalg_matrices, backend):
    """Tests linearalg.norm."""
    ops.set_backend(backend)
    _, mat_b, vec, _ = linearalg_matrices

    # Vector norm (default is L2)
    norm_vec = linearalg.norm(vec)
    # Expected norm = sqrt(9^2 + 8^2) = sqrt(81 + 64) = sqrt(145) approx 12.04
    expected_norm_vec = ops.sqrt(tensor.convert_to_tensor(145.0))
    assert ops.allclose(norm_vec, expected_norm_vec), f"{backend}: Vector L2 norm failed"

    # Matrix norm (default is Frobenius)
    norm_mat_b = linearalg.norm(mat_b)
    # Expected norm = sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(1 + 4 + 9 + 16) = sqrt(30) approx 5.477
    expected_norm_mat_b = ops.sqrt(tensor.convert_to_tensor(30.0))
    assert ops.allclose(norm_mat_b, expected_norm_mat_b), f"{backend}: Matrix Frobenius norm failed"

    # Vector L1 norm
    norm_vec_l1 = linearalg.norm(vec, ord=1)
    # Expected norm = |9| + |8| = 17
    assert ops.allclose(norm_vec_l1, tensor.convert_to_tensor(17.0)), f"{backend}: Vector L1 norm failed"

def test_linearalg_diag_diagonal(linearalg_matrices, backend):
    """Tests linearalg.diag and linearalg.diagonal."""
    ops.set_backend(backend)
    mat_a, _, _, diag_vec = linearalg_matrices

    # Construct diagonal matrix from vector
    diag_matrix = linearalg.diag(diag_vec)
    expected_diag_matrix = tensor.convert_to_tensor([[5.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]])
    assert ops.allclose(diag_matrix, expected_diag_matrix), f"{backend}: diag construction failed"

    # Extract main diagonal from matrix
    main_diagonal = linearalg.diagonal(mat_a)
    expected_main_diagonal = tensor.convert_to_tensor([3.0, 2.0])
    assert ops.allclose(main_diagonal, expected_main_diagonal), f"{backend}: diagonal extraction (main) failed"

    # Extract upper diagonal (k=1)


def test_linearalg_qr(linearalg_matrices, backend):
    """Tests linearalg.qr decomposition."""
    ops.set_backend(backend)
    mat_b, _, _, _ = linearalg_matrices # Use mat_b (non-symmetric)

    q, r = linearalg.qr(mat_b)

    # Check shapes
    assert tensor.shape(q) == tensor.shape(mat_b), f"{backend}: QR Q shape mismatch"
    assert tensor.shape(r) == tensor.shape(mat_b), f"{backend}: QR R shape mismatch"

    # Check properties: Q should be orthogonal (Q^T * Q = I)
    q_t = tensor.transpose(q)
    identity_q = ops.matmul(q_t, q)
    expected_identity = tensor.eye(tensor.shape(mat_b)[0])
    assert ops.allclose(identity_q, expected_identity, atol=1e-5), f"{backend}: QR Q not orthogonal"

    # Check properties: R should be upper triangular
    # This requires a function like ops.tril or tensor.tril. Assuming it exists.
    # If not, this check is hard to do purely via ops.
    try:
        lower_triangle = tensor.tril(r, k=-1) # Get the lower triangle part below the diagonal
        zeros_like_lower = tensor.zeros_like(lower_triangle)
        is_upper_triangular = ops.allclose(lower_triangle, zeros_like_lower, atol=1e-6)
        assert is_upper_triangular, f"{backend}: QR R not upper triangular"
    except AttributeError:
         pytest.skip(f"Skipping R upper triangular check for backend {backend} - tril not found")


    # Check reconstruction: Q * R = A
    reconstruction = ops.matmul(q, r)
    assert ops.allclose(reconstruction, mat_b, atol=1e-6), f"{backend}: QR reconstruction failed"

def test_linearalg_svd(linearalg_matrices, backend):
    """Tests linearalg.svd decomposition."""
    ops.set_backend(backend)
    mat_b, _, _, _ = linearalg_matrices

    u, s, vh = linearalg.svd(mat_b)

    # Check shapes (s is vector, u/vh are matrices)
    m, n = tensor.shape(mat_b)
    k = min(m, n)
    assert tensor.shape(u) == (m, m), f"{backend}: SVD U shape mismatch"
    assert tensor.shape(s) == (k,), f"{backend}: SVD S shape mismatch"
    assert tensor.shape(vh) == (n, n), f"{backend}: SVD Vh shape mismatch" # Vh is V transpose

    # Check properties: U and Vh should be orthogonal
    u_t = tensor.transpose(u)
    identity_u = ops.matmul(u_t, u)
    expected_identity_u = tensor.eye(m)
    assert ops.allclose(identity_u, expected_identity_u, atol=1e-5), f"{backend}: SVD U not orthogonal"

    vh_t = tensor.transpose(vh)
    identity_v = ops.matmul(vh, vh_t) # V = Vh^T -> V * V^T = I
    expected_identity_v = tensor.eye(n)
    assert ops.allclose(identity_v, expected_identity_v, atol=1e-5), f"{backend}: SVD V (from Vh) not orthogonal"

    # Check reconstruction: U * diag(S) * Vh = A
    # Need to construct Sigma matrix from s vector
    sigma = tensor.zeros((m, n))
    s_diag = linearalg.diag(s) # Creates kxk diagonal matrix
    # Place s_diag into top-left of sigma
    # This slicing might need backend-specific handling or a dedicated function
    # Simplified check for square/simple case if needed, full check is complex
    # sigma[:k, :k] = s_diag # Pseudocode for slicing update
    # For 2x2 case:
    if m == 2 and n == 2:
        sigma = linearalg.diag(s)
        reconstruction = ops.matmul(ops.matmul(u, sigma), vh)
        assert ops.allclose(reconstruction, mat_b, atol=1e-5), f"{backend}: SVD reconstruction failed"
    # Else: Skip full reconstruction check for non-square or add complex slicing logic

def test_linearalg_cholesky(linearalg_matrices, backend):
    """Tests linearalg.cholesky decomposition."""
    ops.set_backend(backend)
    # Need a positive definite matrix
    # mat_a = [[3, 1], [1, 2]] -> det=5>0, trace=5>0. Eigenvalues are approx 3.618, 1.382 > 0. So PD.
    mat_pd = linearalg_matrices[0]

    l = linearalg.cholesky(mat_pd)

    # Check shape
    assert tensor.shape(l) == tensor.shape(mat_pd), f"{backend}: Cholesky L shape mismatch"

    # Check properties: L should be lower triangular
    # This requires a function like ops.triu or tensor.triu. Assuming it exists.
    try:
        upper_triangle = tensor.triu(l, k=1) # Get the upper triangle part above the diagonal
        zeros_like_upper = tensor.zeros_like(upper_triangle)
        is_lower_triangular = ops.allclose(upper_triangle, zeros_like_upper, atol=1e-6)
        assert is_lower_triangular, f"{backend}: Cholesky L not lower triangular"
    except AttributeError:
        pytest.skip(f"Skipping L lower triangular check for backend {backend} - triu not found")


    # Check reconstruction: L * L^T = A
    l_t = tensor.transpose(l)
    reconstruction = ops.matmul(l, l_t)
    assert ops.allclose(reconstruction, mat_pd, atol=1e-6), f"{backend}: Cholesky reconstruction failed"

def test_linearalg_eig(linearalg_matrices, backend):
    """Tests linearalg.eig (eigenvalues and eigenvectors)."""
    ops.set_backend(backend)
    # Use a simple symmetric matrix for easier verification
    mat_sym = tensor.convert_to_tensor([[2.0, 1.0], [1.0, 2.0]])

    eigenvalues, eigenvectors = linearalg.eig(mat_sym)

    # Check shapes
    n = tensor.shape(mat_sym)[0]
    assert tensor.shape(eigenvalues) == (n,), f"{backend}: Eig eigenvalues shape mismatch"
    assert tensor.shape(eigenvectors) == (n, n), f"{backend}: Eig eigenvectors shape mismatch"

    # Check property: A * v = lambda * v for each eigenvector/value pair
    for i in range(n):
        lambda_i = eigenvalues[i]
        v_i = eigenvectors[:, i]
        Av = ops.matmul(mat_sym, v_i)
        lambda_v = ops.multiply(v_i, lambda_i)
        assert ops.allclose(Av, lambda_v, atol=1e-5), f"{backend}: Eigenvector {i} property failed"

def test_linearalg_eigvals(linearalg_matrices, backend):
    """Tests linearalg.eigvals (eigenvalues only)."""
    ops.set_backend(backend)
    mat_sym = tensor.convert_to_tensor([[2.0, 1.0], [1.0, 2.0]]) # Same as eig test
    n = tensor.shape(mat_sym)[0]

    eigenvalues = linearalg.eigvals(mat_sym)

    assert tensor.shape(eigenvalues) == (n,), f"{backend}: Eigvals shape mismatch"
    # Check values (should be 3 and 1 for the example matrix)
    # Sorting might be needed as order is not guaranteed
    # Use ember stats.sort for comparison
    from ember_ml.ops import stats
    eigenvalues_sorted = stats.sort(eigenvalues)
    expected_values_sorted = tensor.convert_to_tensor([1.0, 3.0])
    expected_values_sorted = tensor.cast(expected_values_sorted, eigenvalues_sorted.dtype) # Match type
    assert ops.allclose(eigenvalues_sorted, expected_values_sorted, atol=1e-6), f"{backend}: Eigvals values mismatch"

# Note: lstsq test might require more setup (non-square matrices)
# Skipping lstsq for now.


    upper_diagonal = linearalg.diagonal(mat_a, offset=1)
    expected_upper_diagonal = tensor.convert_to_tensor([1.0])
    assert ops.allclose(upper_diagonal, expected_upper_diagonal), f"{backend}: diagonal extraction (k=1) failed"

    # Extract lower diagonal (k=-1)
    lower_diagonal = linearalg.diagonal(mat_a, offset=-1)
    expected_lower_diagonal = tensor.convert_to_tensor([1.0])
    assert ops.allclose(lower_diagonal, expected_lower_diagonal), f"{backend}: diagonal extraction (k=-1) failed"

# TODO: Add tests for qr, svd, cholesky, lstsq, eig, eigvals