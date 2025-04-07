import pytest
from pytest import mark
from ember_ml import ops
from ember_ml.ops import linearalg # Import the linearalg submodule
from ember_ml.nn import tensor

# Define the backend order: numpy -> torch -> mlx

# Helper function to get linear algebra matrices
def _get_linearalg_matrices():
    mat_a = tensor.convert_to_tensor([[3.0, 1.0], [1.0, 2.0]]) # Square invertible
    mat_b = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]]) # Another matrix
    vec = tensor.convert_to_tensor([9.0, 8.0]) # Vector for solving
    diag_vec = tensor.convert_to_tensor([5.0, 6.0, 7.0]) # Vector for diagonal
    mat_sym = tensor.convert_to_tensor([[2.0, 1.0], [1.0, 2.0]]) # Symmetric for eig tests
    mat_pd = mat_a # A is positive definite
    return mat_a, mat_b, vec, diag_vec, mat_sym, mat_pd

# --- NumPy Backend Setup & Tests ---
@mark.run(order=1)
def test_setup_numpy():
    """Set up the NumPy backend."""
    print("\n=== Setting backend to NumPy ===")
    ops.set_backend('numpy')
    assert ops.get_backend() == 'numpy'

@mark.run(order=1)
def test_linearalg_solve_numpy():
    """Tests linearalg.solve with NumPy backend."""
    mat_a, _, vec, _, _, _ = _get_linearalg_matrices()
    solution = linearalg.solve(mat_a, vec)
    expected_solution = tensor.convert_to_tensor([2.0, 3.0])
    assert ops.allclose(solution, expected_solution, atol=1e-6), "Solve failed"
    verification = ops.matmul(mat_a, solution)
    assert ops.allclose(verification, vec), "Solve verification failed"

@mark.run(order=1)
def test_linearalg_inv_numpy():
    """Tests linearalg.inv with NumPy backend."""
    mat_a, _, _, _, _, _ = _get_linearalg_matrices()
    inverse_a = linearalg.inv(mat_a)
    expected_inverse_a = tensor.convert_to_tensor([[0.4, -0.2], [-0.2, 0.6]])
    assert ops.allclose(inverse_a, expected_inverse_a, atol=1e-6), "Inverse calculation failed"
    identity_calc = ops.matmul(mat_a, inverse_a)
    n_rows = tensor.shape(mat_a)[0]
    expected_identity = tensor.eye(n_rows)
    expected_identity = tensor.cast(expected_identity, identity_calc.dtype)
    assert ops.allclose(identity_calc, expected_identity, atol=1e-6), "Inverse verification failed"

@mark.run(order=1)
def test_linearalg_det_numpy():
    """Tests linearalg.det with NumPy backend."""
    mat_a, mat_b, _, _, _, _ = _get_linearalg_matrices()
    det_a = linearalg.det(mat_a)
    assert ops.allclose(det_a, tensor.convert_to_tensor(5.0)), "Determinant(A) failed"
    det_b = linearalg.det(mat_b)
    assert ops.allclose(det_b, tensor.convert_to_tensor(-2.0)), "Determinant(B) failed"

@mark.run(order=1)
def test_linearalg_norm_numpy():
    """Tests linearalg.norm with NumPy backend."""
    _, mat_b, vec, _, _, _ = _get_linearalg_matrices()
    norm_vec = linearalg.norm(vec)
    expected_norm_vec = ops.sqrt(tensor.convert_to_tensor(145.0))
    assert ops.allclose(norm_vec, expected_norm_vec), "Vector L2 norm failed"
    norm_mat_b = linearalg.norm(mat_b)
    expected_norm_mat_b = ops.sqrt(tensor.convert_to_tensor(30.0))
    assert ops.allclose(norm_mat_b, expected_norm_mat_b), "Matrix Frobenius norm failed"
    norm_vec_l1 = linearalg.norm(vec, ord=1)
    assert ops.allclose(norm_vec_l1, tensor.convert_to_tensor(17.0)), "Vector L1 norm failed"

@mark.run(order=1)
def test_linearalg_diag_diagonal_numpy():
    """Tests linearalg.diag and linearalg.diagonal with NumPy backend."""
    mat_a, _, _, diag_vec, _, _ = _get_linearalg_matrices()
    diag_matrix = linearalg.diag(diag_vec)
    expected_diag_matrix = tensor.convert_to_tensor([[5.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]])
    assert ops.allclose(diag_matrix, expected_diag_matrix), "diag construction failed"
    main_diagonal = linearalg.diagonal(mat_a)
    expected_main_diagonal = tensor.convert_to_tensor([3.0, 2.0])
    assert ops.allclose(main_diagonal, expected_main_diagonal), "diagonal extraction (main) failed"
    upper_diagonal = linearalg.diagonal(mat_a, offset=1)
    expected_upper_diagonal = tensor.convert_to_tensor([1.0])
    assert ops.allclose(upper_diagonal, expected_upper_diagonal), "diagonal extraction (k=1) failed"
    lower_diagonal = linearalg.diagonal(mat_a, offset=-1)
    expected_lower_diagonal = tensor.convert_to_tensor([1.0])
    assert ops.allclose(lower_diagonal, expected_lower_diagonal), "diagonal extraction (k=-1) failed"

@mark.run(order=1)
def test_linearalg_qr_numpy():
    """Tests linearalg.qr decomposition with NumPy backend."""
    _, mat_b, _, _, _, _ = _get_linearalg_matrices()
    q, r = linearalg.qr(mat_b)
    assert tensor.shape(q) == tensor.shape(mat_b), "QR Q shape mismatch"
    assert tensor.shape(r) == tensor.shape(mat_b), "QR R shape mismatch"
    q_t = tensor.transpose(q)
    identity_q = ops.matmul(q_t, q)
    expected_identity = tensor.eye(tensor.shape(mat_b)[0])
    assert ops.allclose(identity_q, expected_identity, atol=1e-5), "QR Q not orthogonal"
    try:
        lower_triangle = tensor.tril(r, k=-1)
        zeros_like_lower = tensor.zeros_like(lower_triangle)
        is_upper_triangular = ops.allclose(lower_triangle, zeros_like_lower, atol=1e-6)
        assert is_upper_triangular, "QR R not upper triangular"
    except AttributeError: pytest.skip("tril not found")
    reconstruction = ops.matmul(q, r)
    assert ops.allclose(reconstruction, mat_b, atol=1e-6), "QR reconstruction failed"

@mark.run(order=1)
def test_linearalg_svd_numpy():
    """Tests linearalg.svd decomposition with NumPy backend."""
    _, mat_b, _, _, _, _ = _get_linearalg_matrices()
    u, s, vh = linearalg.svd(mat_b)
    m, n = tensor.shape(mat_b); k = min(m, n)
    assert tensor.shape(u) == (m, m), "SVD U shape mismatch"
    assert tensor.shape(s) == (k,), "SVD S shape mismatch"
    assert tensor.shape(vh) == (n, n), "SVD Vh shape mismatch"
    u_t = tensor.transpose(u); identity_u = ops.matmul(u_t, u)
    expected_identity_u = tensor.eye(m)
    assert ops.allclose(identity_u, expected_identity_u, atol=1e-5), "SVD U not orthogonal"
    vh_t = tensor.transpose(vh); identity_v = ops.matmul(vh, vh_t)
    expected_identity_v = tensor.eye(n)
    assert ops.allclose(identity_v, expected_identity_v, atol=1e-5), "SVD V not orthogonal"
    if m == 2 and n == 2: # Simplified check for 2x2 case
        sigma = linearalg.diag(s)
        reconstruction = ops.matmul(ops.matmul(u, sigma), vh)
        assert ops.allclose(reconstruction, mat_b, atol=1e-5), "SVD reconstruction failed"

@mark.run(order=1)
def test_linearalg_cholesky_numpy():
    """Tests linearalg.cholesky decomposition with NumPy backend."""
    _, _, _, _, _, mat_pd = _get_linearalg_matrices()
    l = linearalg.cholesky(mat_pd)
    assert tensor.shape(l) == tensor.shape(mat_pd), "Cholesky L shape mismatch"
    try:
        upper_triangle = tensor.triu(l, k=1)
        zeros_like_upper = tensor.zeros_like(upper_triangle)
        is_lower_triangular = ops.allclose(upper_triangle, zeros_like_upper, atol=1e-6)
        assert is_lower_triangular, "Cholesky L not lower triangular"
    except AttributeError: pytest.skip("triu not found")
    l_t = tensor.transpose(l)
    reconstruction = ops.matmul(l, l_t)
    assert ops.allclose(reconstruction, mat_pd, atol=1e-6), "Cholesky reconstruction failed"

@mark.run(order=1)
def test_linearalg_eig_numpy():
    """Tests linearalg.eig with NumPy backend."""
    _, _, _, _, mat_sym, _ = _get_linearalg_matrices()
    eigenvalues, eigenvectors = linearalg.eig(mat_sym)
    n = tensor.shape(mat_sym)[0]
    assert tensor.shape(eigenvalues) == (n,), "Eig eigenvalues shape mismatch"
    assert tensor.shape(eigenvectors) == (n, n), "Eig eigenvectors shape mismatch"
    for i in range(n):
        lambda_i = eigenvalues[i]
        v_i = eigenvectors[:, i]
        Av = ops.matmul(mat_sym, v_i)
        lambda_v = ops.multiply(v_i, lambda_i)
        assert ops.allclose(Av, lambda_v, atol=1e-5), f"Eigenvector {i} property failed"

@mark.run(order=1)
def test_linearalg_eigvals_numpy():
    """Tests linearalg.eigvals with NumPy backend."""
    from ember_ml.ops import stats # Import locally
    _, _, _, _, mat_sym, _ = _get_linearalg_matrices()
    n = tensor.shape(mat_sym)[0]
    eigenvalues = linearalg.eigvals(mat_sym)
    assert tensor.shape(eigenvalues) == (n,), "Eigvals shape mismatch"
    eigenvalues_sorted = stats.sort(eigenvalues)
    expected_values_sorted = tensor.convert_to_tensor([1.0, 3.0])
    expected_values_sorted = tensor.cast(expected_values_sorted, eigenvalues_sorted.dtype)
    assert ops.allclose(eigenvalues_sorted, expected_values_sorted, atol=1e-6), "Eigvals values mismatch"

# --- PyTorch Backend Setup & Tests ---
@mark.run(order=2)
def test_setup_torch():
    """Set up the PyTorch backend."""
    print("\n=== Setting backend to PyTorch ===")
    try:
        import torch
        ops.set_backend('torch')
        assert ops.get_backend() == 'torch'
    except ImportError:
        pytest.skip("PyTorch not available")

@mark.run(order=2)
def test_linearalg_solve_torch():
    """Tests linearalg.solve with PyTorch backend."""
    mat_a, _, vec, _, _, _ = _get_linearalg_matrices()
    solution = linearalg.solve(mat_a, vec)
    expected_solution = tensor.convert_to_tensor([2.0, 3.0])
    assert ops.allclose(solution, expected_solution, atol=1e-6), "Solve failed"
    verification = ops.matmul(mat_a, solution)
    assert ops.allclose(verification, vec), "Solve verification failed"

@mark.run(order=2)
def test_linearalg_inv_torch():
    """Tests linearalg.inv with PyTorch backend."""
    mat_a, _, _, _, _, _ = _get_linearalg_matrices()
    inverse_a = linearalg.inv(mat_a)
    expected_inverse_a = tensor.convert_to_tensor([[0.4, -0.2], [-0.2, 0.6]])
    assert ops.allclose(inverse_a, expected_inverse_a, atol=1e-6), "Inverse calculation failed"
    identity_calc = ops.matmul(mat_a, inverse_a)
    n_rows = tensor.shape(mat_a)[0]
    expected_identity = tensor.eye(n_rows)
    expected_identity = tensor.cast(expected_identity, identity_calc.dtype)
    assert ops.allclose(identity_calc, expected_identity, atol=1e-6), "Inverse verification failed"

@mark.run(order=2)
def test_linearalg_det_torch():
    """Tests linearalg.det with PyTorch backend."""
    mat_a, mat_b, _, _, _, _ = _get_linearalg_matrices()
    det_a = linearalg.det(mat_a)
    assert ops.allclose(det_a, tensor.convert_to_tensor(5.0)), "Determinant(A) failed"
    det_b = linearalg.det(mat_b)
    assert ops.allclose(det_b, tensor.convert_to_tensor(-2.0)), "Determinant(B) failed"

@mark.run(order=2)
def test_linearalg_norm_torch():
    """Tests linearalg.norm with PyTorch backend."""
    _, mat_b, vec, _, _, _ = _get_linearalg_matrices()
    norm_vec = linearalg.norm(vec)
    expected_norm_vec = ops.sqrt(tensor.convert_to_tensor(145.0))
    assert ops.allclose(norm_vec, expected_norm_vec), "Vector L2 norm failed"
    norm_mat_b = linearalg.norm(mat_b)
    expected_norm_mat_b = ops.sqrt(tensor.convert_to_tensor(30.0))
    assert ops.allclose(norm_mat_b, expected_norm_mat_b), "Matrix Frobenius norm failed"
    norm_vec_l1 = linearalg.norm(vec, ord=1)
    assert ops.allclose(norm_vec_l1, tensor.convert_to_tensor(17.0)), "Vector L1 norm failed"

@mark.run(order=2)
def test_linearalg_diag_diagonal_torch():
    """Tests linearalg.diag and linearalg.diagonal with PyTorch backend."""
    mat_a, _, _, diag_vec, _, _ = _get_linearalg_matrices()
    diag_matrix = linearalg.diag(diag_vec)
    expected_diag_matrix = tensor.convert_to_tensor([[5.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]])
    assert ops.allclose(diag_matrix, expected_diag_matrix), "diag construction failed"
    main_diagonal = linearalg.diagonal(mat_a)
    expected_main_diagonal = tensor.convert_to_tensor([3.0, 2.0])
    assert ops.allclose(main_diagonal, expected_main_diagonal), "diagonal extraction (main) failed"
    upper_diagonal = linearalg.diagonal(mat_a, offset=1)
    expected_upper_diagonal = tensor.convert_to_tensor([1.0])
    assert ops.allclose(upper_diagonal, expected_upper_diagonal), "diagonal extraction (k=1) failed"
    lower_diagonal = linearalg.diagonal(mat_a, offset=-1)
    expected_lower_diagonal = tensor.convert_to_tensor([1.0])
    assert ops.allclose(lower_diagonal, expected_lower_diagonal), "diagonal extraction (k=-1) failed"

@mark.run(order=2)
def test_linearalg_qr_torch():
    """Tests linearalg.qr decomposition with PyTorch backend."""
    _, mat_b, _, _, _, _ = _get_linearalg_matrices()
    q, r = linearalg.qr(mat_b)
    assert tensor.shape(q) == tensor.shape(mat_b), "QR Q shape mismatch"
    assert tensor.shape(r) == tensor.shape(mat_b), "QR R shape mismatch"
    q_t = tensor.transpose(q)
    identity_q = ops.matmul(q_t, q)
    expected_identity = tensor.eye(tensor.shape(mat_b)[0])
    assert ops.allclose(identity_q, expected_identity, atol=1e-5), "QR Q not orthogonal"
    try:
        lower_triangle = tensor.tril(r, k=-1)
        zeros_like_lower = tensor.zeros_like(lower_triangle)
        is_upper_triangular = ops.allclose(lower_triangle, zeros_like_lower, atol=1e-6)
        assert is_upper_triangular, "QR R not upper triangular"
    except AttributeError: pytest.skip("tril not found")
    reconstruction = ops.matmul(q, r)
    assert ops.allclose(reconstruction, mat_b, atol=1e-6), "QR reconstruction failed"

@mark.run(order=2)
def test_linearalg_svd_torch():
    """Tests linearalg.svd decomposition with PyTorch backend."""
    _, mat_b, _, _, _, _ = _get_linearalg_matrices()
    u, s, vh = linearalg.svd(mat_b)
    m, n = tensor.shape(mat_b); k = min(m, n)
    assert tensor.shape(u) == (m, m), "SVD U shape mismatch"
    assert tensor.shape(s) == (k,), "SVD S shape mismatch"
    assert tensor.shape(vh) == (n, n), "SVD Vh shape mismatch"
    u_t = tensor.transpose(u); identity_u = ops.matmul(u_t, u)
    expected_identity_u = tensor.eye(m)
    assert ops.allclose(identity_u, expected_identity_u, atol=1e-5), "SVD U not orthogonal"
    vh_t = tensor.transpose(vh); identity_v = ops.matmul(vh, vh_t)
    expected_identity_v = tensor.eye(n)
    assert ops.allclose(identity_v, expected_identity_v, atol=1e-5), "SVD V not orthogonal"
    if m == 2 and n == 2:
        sigma = linearalg.diag(s)
        reconstruction = ops.matmul(ops.matmul(u, sigma), vh)
        assert ops.allclose(reconstruction, mat_b, atol=1e-5), "SVD reconstruction failed"

@mark.run(order=2)
def test_linearalg_cholesky_torch():
    """Tests linearalg.cholesky decomposition with PyTorch backend."""
    _, _, _, _, _, mat_pd = _get_linearalg_matrices()
    l = linearalg.cholesky(mat_pd)
    assert tensor.shape(l) == tensor.shape(mat_pd), "Cholesky L shape mismatch"
    try:
        upper_triangle = tensor.triu(l, k=1)
        zeros_like_upper = tensor.zeros_like(upper_triangle)
        is_lower_triangular = ops.allclose(upper_triangle, zeros_like_upper, atol=1e-6)
        assert is_lower_triangular, "Cholesky L not lower triangular"
    except AttributeError: pytest.skip("triu not found")
    l_t = tensor.transpose(l)
    reconstruction = ops.matmul(l, l_t)
    assert ops.allclose(reconstruction, mat_pd, atol=1e-6), "Cholesky reconstruction failed"

@mark.run(order=2)
def test_linearalg_eig_torch():
    """Tests linearalg.eig with PyTorch backend."""
    _, _, _, _, mat_sym, _ = _get_linearalg_matrices()
    eigenvalues, eigenvectors = linearalg.eig(mat_sym)
    n = tensor.shape(mat_sym)[0]
    assert tensor.shape(eigenvalues) == (n,), "Eig eigenvalues shape mismatch"
    assert tensor.shape(eigenvectors) == (n, n), "Eig eigenvectors shape mismatch"
    for i in range(n):
        lambda_i = eigenvalues[i]
        v_i = eigenvectors[:, i]
        Av = ops.matmul(mat_sym, v_i)
        lambda_v = ops.multiply(v_i, lambda_i)
        assert ops.allclose(Av, lambda_v, atol=1e-5), f"Eigenvector {i} property failed"

@mark.run(order=2)
def test_linearalg_eigvals_torch():
    """Tests linearalg.eigvals with PyTorch backend."""
    from ember_ml.ops import stats # Import locally
    _, _, _, _, mat_sym, _ = _get_linearalg_matrices()
    n = tensor.shape(mat_sym)[0]
    eigenvalues = linearalg.eigvals(mat_sym)
    assert tensor.shape(eigenvalues) == (n,), "Eigvals shape mismatch"
    eigenvalues_sorted = stats.sort(eigenvalues)
    expected_values_sorted = tensor.convert_to_tensor([1.0, 3.0])
    expected_values_sorted = tensor.cast(expected_values_sorted, eigenvalues_sorted.dtype)
    assert ops.allclose(eigenvalues_sorted, expected_values_sorted, atol=1e-6), "Eigvals values mismatch"

# --- MLX Backend Setup & Tests ---
@mark.run(order=3)
def test_setup_mlx():
    """Set up the MLX backend."""
    print("\n=== Setting backend to MLX ===")
    try:
        import mlx.core
        ops.set_backend('mlx')
        assert ops.get_backend() == 'mlx'
    except ImportError:
        pytest.skip("MLX not available")

@mark.run(order=3)
def test_linearalg_solve_mlx():
    """Tests linearalg.solve with MLX backend."""
    mat_a, _, vec, _, _, _ = _get_linearalg_matrices()
    solution = linearalg.solve(mat_a, vec)
    expected_solution = tensor.convert_to_tensor([2.0, 3.0])
    assert ops.allclose(solution, expected_solution, atol=1e-6), "Solve failed"
    verification = ops.matmul(mat_a, solution)
    assert ops.allclose(verification, vec), "Solve verification failed"

@mark.run(order=3)
def test_linearalg_inv_mlx():
    """Tests linearalg.inv with MLX backend."""
    mat_a, _, _, _, _, _ = _get_linearalg_matrices()
    inverse_a = linearalg.inv(mat_a)
    expected_inverse_a = tensor.convert_to_tensor([[0.4, -0.2], [-0.2, 0.6]])
    assert ops.allclose(inverse_a, expected_inverse_a, atol=1e-6), "Inverse calculation failed"
    identity_calc = ops.matmul(mat_a, inverse_a)
    n_rows = tensor.shape(mat_a)[0]
    expected_identity = tensor.eye(n_rows)
    expected_identity = tensor.cast(expected_identity, identity_calc.dtype)
    assert ops.allclose(identity_calc, expected_identity, atol=1e-6), "Inverse verification failed"

@mark.run(order=3)
def test_linearalg_det_mlx():
    """Tests linearalg.det with MLX backend."""
    mat_a, mat_b, _, _, _, _ = _get_linearalg_matrices()
    det_a = linearalg.det(mat_a)
    assert ops.allclose(det_a, tensor.convert_to_tensor(5.0)), "Determinant(A) failed"
    det_b = linearalg.det(mat_b)
    assert ops.allclose(det_b, tensor.convert_to_tensor(-2.0)), "Determinant(B) failed"

@mark.run(order=3)
def test_linearalg_norm_mlx():
    """Tests linearalg.norm with MLX backend."""
    _, mat_b, vec, _, _, _ = _get_linearalg_matrices()
    norm_vec = linearalg.norm(vec)
    expected_norm_vec = ops.sqrt(tensor.convert_to_tensor(145.0))
    assert ops.allclose(norm_vec, expected_norm_vec), "Vector L2 norm failed"
    norm_mat_b = linearalg.norm(mat_b)
    expected_norm_mat_b = ops.sqrt(tensor.convert_to_tensor(30.0))
    assert ops.allclose(norm_mat_b, expected_norm_mat_b), "Matrix Frobenius norm failed"
    norm_vec_l1 = linearalg.norm(vec, ord=1)
    assert ops.allclose(norm_vec_l1, tensor.convert_to_tensor(17.0)), "Vector L1 norm failed"

@mark.run(order=3)
def test_linearalg_diag_diagonal_mlx():
    """Tests linearalg.diag and linearalg.diagonal with MLX backend."""
    mat_a, _, _, diag_vec, _, _ = _get_linearalg_matrices()
    diag_matrix = linearalg.diag(diag_vec)
    expected_diag_matrix = tensor.convert_to_tensor([[5.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]])
    assert ops.allclose(diag_matrix, expected_diag_matrix), "diag construction failed"
    main_diagonal = linearalg.diagonal(mat_a)
    expected_main_diagonal = tensor.convert_to_tensor([3.0, 2.0])
    assert ops.allclose(main_diagonal, expected_main_diagonal), "diagonal extraction (main) failed"
    upper_diagonal = linearalg.diagonal(mat_a, offset=1)
    expected_upper_diagonal = tensor.convert_to_tensor([1.0])
    assert ops.allclose(upper_diagonal, expected_upper_diagonal), "diagonal extraction (k=1) failed"
    lower_diagonal = linearalg.diagonal(mat_a, offset=-1)
    expected_lower_diagonal = tensor.convert_to_tensor([1.0])
    assert ops.allclose(lower_diagonal, expected_lower_diagonal), "diagonal extraction (k=-1) failed"

@mark.run(order=3)
def test_linearalg_qr_mlx():
    """Tests linearalg.qr decomposition with MLX backend."""
    _, mat_b, _, _, _, _ = _get_linearalg_matrices()
    q, r = linearalg.qr(mat_b)
    assert tensor.shape(q) == tensor.shape(mat_b), "QR Q shape mismatch"
    assert tensor.shape(r) == tensor.shape(mat_b), "QR R shape mismatch"
    q_t = tensor.transpose(q)
    identity_q = ops.matmul(q_t, q)
    expected_identity = tensor.eye(tensor.shape(mat_b)[0])
    assert ops.allclose(identity_q, expected_identity, atol=1e-5), "QR Q not orthogonal"
    try:
        lower_triangle = tensor.tril(r, k=-1)
        zeros_like_lower = tensor.zeros_like(lower_triangle)
        is_upper_triangular = ops.allclose(lower_triangle, zeros_like_lower, atol=1e-6)
        assert is_upper_triangular, "QR R not upper triangular"
    except AttributeError: pytest.skip("tril not found")
    reconstruction = ops.matmul(q, r)
    assert ops.allclose(reconstruction, mat_b, atol=1e-6), "QR reconstruction failed"

@mark.run(order=3)
def test_linearalg_svd_mlx():
    """Tests linearalg.svd decomposition with MLX backend."""
    _, mat_b, _, _, _, _ = _get_linearalg_matrices()
    u, s, vh = linearalg.svd(mat_b)
    m, n = tensor.shape(mat_b); k = min(m, n)
    assert tensor.shape(u) == (m, m), "SVD U shape mismatch"
    assert tensor.shape(s) == (k,), "SVD S shape mismatch"
    assert tensor.shape(vh) == (n, n), "SVD Vh shape mismatch"
    u_t = tensor.transpose(u); identity_u = ops.matmul(u_t, u)
    expected_identity_u = tensor.eye(m)
    assert ops.allclose(identity_u, expected_identity_u, atol=1e-5), "SVD U not orthogonal"
    vh_t = tensor.transpose(vh); identity_v = ops.matmul(vh, vh_t)
    expected_identity_v = tensor.eye(n)
    assert ops.allclose(identity_v, expected_identity_v, atol=1e-5), "SVD V not orthogonal"
    if m == 2 and n == 2:
        sigma = linearalg.diag(s)
        reconstruction = ops.matmul(ops.matmul(u, sigma), vh)
        assert ops.allclose(reconstruction, mat_b, atol=1e-5), "SVD reconstruction failed"

@mark.run(order=3)
def test_linearalg_cholesky_mlx():
    """Tests linearalg.cholesky decomposition with MLX backend."""
    _, _, _, _, _, mat_pd = _get_linearalg_matrices()
    l = linearalg.cholesky(mat_pd)
    assert tensor.shape(l) == tensor.shape(mat_pd), "Cholesky L shape mismatch"
    try:
        upper_triangle = tensor.triu(l, k=1)
        zeros_like_upper = tensor.zeros_like(upper_triangle)
        is_lower_triangular = ops.allclose(upper_triangle, zeros_like_upper, atol=1e-6)
        assert is_lower_triangular, "Cholesky L not lower triangular"
    except AttributeError: pytest.skip("triu not found")
    l_t = tensor.transpose(l)
    reconstruction = ops.matmul(l, l_t)
    assert ops.allclose(reconstruction, mat_pd, atol=1e-6), "Cholesky reconstruction failed"

@mark.run(order=3)
def test_linearalg_eig_mlx():
    """Tests linearalg.eig with MLX backend."""
    _, _, _, _, mat_sym, _ = _get_linearalg_matrices()
    eigenvalues, eigenvectors = linearalg.eig(mat_sym)
    n = tensor.shape(mat_sym)[0]
    assert tensor.shape(eigenvalues) == (n,), "Eig eigenvalues shape mismatch"
    assert tensor.shape(eigenvectors) == (n, n), "Eig eigenvectors shape mismatch"
    for i in range(n):
        lambda_i = eigenvalues[i]
        v_i = eigenvectors[:, i]
        Av = ops.matmul(mat_sym, v_i)
        lambda_v = ops.multiply(v_i, lambda_i)
        assert ops.allclose(Av, lambda_v, atol=1e-5), f"Eigenvector {i} property failed"

@mark.run(order=3)
def test_linearalg_eigvals_mlx():
    """Tests linearalg.eigvals with MLX backend."""
    from ember_ml.ops import stats # Import locally
    _, _, _, _, mat_sym, _ = _get_linearalg_matrices()
    n = tensor.shape(mat_sym)[0]
    eigenvalues = linearalg.eigvals(mat_sym)
    assert tensor.shape(eigenvalues) == (n,), "Eigvals shape mismatch"
    eigenvalues_sorted = stats.sort(eigenvalues)
    expected_values_sorted = tensor.convert_to_tensor([1.0, 3.0])
    expected_values_sorted = tensor.cast(expected_values_sorted, eigenvalues_sorted.dtype)
    assert ops.allclose(eigenvalues_sorted, expected_values_sorted, atol=1e-6), "Eigvals values mismatch"

# Note: lstsq test skipped