"""
Test script for solver operations.

This script tests the solver operations implemented in the NumPy backend.
"""

import importlib
from ember_ml.backend import set_backend
from ember_ml import ops
from ember_ml.nn import tensor
def test_solve():
    """Test the solve operation."""
    # Reset the ops module to clear cached instances
    importlib.reload(ops)
    
    # Set the backend to NumPy
    set_backend("numpy")
    
    # Create a simple linear system
    a = tensor.convert_to_tensor([[3, 1], [1, 2]])
    b = tensor.convert_to_tensor([9, 8])
    
    # Solve the system using our implementation
    x = ops.solve(a, b)
    
    # Expected solution
    x_expected = tensor.convert_to_tensor([2, 3])
    
    # Check that the results are close
    assert ops.allclose(x, x_expected)
    print("solve test passed")

def test_inv():
    """Test the inv operation."""
    # Reset the ops module to clear cached instances
    importlib.reload(ops)
    
    # Set the backend to NumPy
    set_backend("numpy")
    
    # Create a simple matrix
    a = tensor.convert_to_tensor([[3, 1], [1, 2]])
    
    # Compute the inverse using our implementation
    a_inv = ops.inv(a)
    
    # Expected inverse
    a_inv_expected = tensor.convert_to_tensor([[2/5, -1/5], [-1/5, 3/5]])
    
    # Check that the results are close
    assert ops.allclose(a_inv, a_inv_expected)
    print("inv test passed")

def test_det():
    """Test the det operation."""
    # Reset the ops module to clear cached instances
    importlib.reload(ops)
    
    # Set the backend to NumPy
    set_backend("numpy")
    
    # Create a simple matrix
    a = tensor.convert_to_tensor([[3, 1], [1, 2]])
    
    # Compute the determinant using our implementation
    det_a = ops.det(a)
    
    # Expected determinant
    det_a_expected = tensor.convert_to_tensor(5)
    
    # Check that the results are close
    assert ops.isclose(det_a, det_a_expected)
    print("det test passed")

def test_norm():
    """Test the norm operation."""
    # Reset the ops module to clear cached instances
    importlib.reload(ops)
    
    # Set the backend to NumPy
    set_backend("numpy")
    
    # Create a simple matrix
    a = tensor.convert_to_tensor([[3, 1], [1, 2]])
    
    # Compute the norm using our implementation
    norm_a = ops.norm(a)
    
    # Expected norm (Frobenius norm)
    norm_a_expected = tensor.convert_to_tensor(ops.sqrt(3*3 + 1*1 + 1*1 + 2*2))
    
    # Check that the results are close
    assert ops.isclose(norm_a, norm_a_expected)
    print("norm test passed")

def test_qr():
    """Test the qr operation."""
    # Reset the ops module to clear cached instances
    importlib.reload(ops)
    
    # Set the backend to NumPy
    set_backend("numpy")
    
    # Create a simple matrix
    a = tensor.convert_to_tensor([[3, 1], [1, 2], [0, 1]])
    
    # Compute the QR decomposition using our implementation
    q, r = ops.qr(a)
    
    # Check that Q is orthogonal
    q_transpose = ops.transpose(q)
    q_transpose_q = ops.matmul(q_transpose, q)
    identity = ops.eye(q.shape[1])
    assert ops.allclose(q_transpose_q, identity)
    
    # Check that A = QR
    qr_product = ops.matmul(q, r)
    assert ops.allclose(a, qr_product)
    
    print("qr test passed")

def test_svd():
    """Test the svd operation."""
    # Reset the ops module to clear cached instances
    importlib.reload(ops)
    
    # Set the backend to NumPy
    set_backend("numpy")
    
    # Create a simple matrix
    a = tensor.convert_to_tensor([[3, 1], [1, 2], [0, 1]])
    
    # Compute the SVD using our implementation
    u, s, vh = ops.svd(a)
    
    # Check that U is orthogonal
    u_transpose = ops.transpose(u)
    u_transpose_u = ops.matmul(u_transpose, u)
    identity_u = ops.eye(u.shape[1])
    assert ops.allclose(u_transpose_u, identity_u)
    
    # Check that V is orthogonal
    vh_transpose = ops.transpose(vh)
    vh_vh_transpose = ops.matmul(vh, vh_transpose)
    identity_v = ops.eye(vh.shape[0])
    assert ops.allclose(vh_vh_transpose, identity_v)
    
    # Check that A = U * diag(S) * Vh
    s_diag = tensor.zeros((u.shape[1], vh.shape[0]))
    for i in range(len(s)):
        s_diag = ops.tensor_ops().tensor_scatter_nd_update(s_diag, [[i, i]], [s[i]])
    
    usv = ops.matmul(ops.matmul(u, s_diag), vh)
    assert ops.allclose(a, usv)
    
    print("svd test passed")

def test_cholesky():
    """Test the cholesky operation."""
    # Reset the ops module to clear cached instances
    importlib.reload(ops)
    
    # Set the backend to NumPy
    set_backend("numpy")
    
    # Create a positive definite matrix
    a = tensor.convert_to_tensor([[4, 1], [1, 3]])
    
    # Compute the Cholesky decomposition using our implementation
    l = ops.cholesky(a)
    
    # Check that A = L * L^T
    l_transpose = ops.transpose(l)
    ll_transpose = ops.matmul(l, l_transpose)
    assert ops.allclose(a, ll_transpose)
    
    print("cholesky test passed")

def test_lstsq():
    """Test the lstsq operation."""
    # Reset the ops module to clear cached instances
    importlib.reload(ops)
    
    # Set the backend to NumPy
    set_backend("numpy")
    
    # Create an overdetermined system
    a = tensor.convert_to_tensor([[3, 1], [1, 2], [0, 1]])
    b = tensor.convert_to_tensor([9, 8, 3])
    
    # Compute the least-squares solution using our implementation
    x, residuals, rank, s = ops.lstsq(a, b)
    
    # Check that x minimizes ||Ax - b||
    ax = ops.matmul(a, x)
    residual_norm = ops.norm(ops.subtract(ax, b))
    
    # Try a different solution and check that it has a larger residual
    x_bad = ops.add(x, tensor.convert_to_tensor([0.1, 0.1]))
    ax_bad = ops.matmul(a, x_bad)
    residual_norm_bad = ops.norm(ops.subtract(ax_bad, b))
    
    assert residual_norm < residual_norm_bad
    
    print("lstsq test passed")

def test_eig():
    """Test the eig operation."""
    # Reset the ops module to clear cached instances
    importlib.reload(ops)
    
    # Set the backend to NumPy
    set_backend("numpy")
    
    # Create a simple matrix
    a = tensor.convert_to_tensor([[3, 1], [1, 2]])
    
    # Compute the eigenvalues and eigenvectors using our implementation
    eigenvalues, eigenvectors = ops.eig(a)
    
    # Check that A * v = lambda * v for each eigenvector v and eigenvalue lambda
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lambda_v = ops.multiply(eigenvalues[i], v)
        av = ops.matmul(a, v)
        assert ops.allclose(av, lambda_v)
    
    print("eig test passed")

def test_eigvals():
    """Test the eigvals operation."""
    # Reset the ops module to clear cached instances
    importlib.reload(ops)
    
    # Set the backend to NumPy
    set_backend("numpy")
    
    # Create a simple matrix
    a = tensor.convert_to_tensor([[3, 1], [1, 2]])
    
    # Compute the eigenvalues using our implementation
    eigenvalues = ops.eigvals(a)
    
    # Compute the eigenvalues using eig
    eigenvalues_from_eig, _ = ops.eig(a)
    
    # Check that the results are close
    assert ops.allclose(ops.sort(eigenvalues), ops.sort(eigenvalues_from_eig))
    
    print("eigvals test passed")

if __name__ == "__main__":
    # Run all tests
    test_solve()
    test_inv()
    test_det()
    test_norm()
    test_qr()
    test_svd()
    test_cholesky()
    test_lstsq()
    test_eig()
    test_eigvals()
    
    print("\nAll tests passed!")