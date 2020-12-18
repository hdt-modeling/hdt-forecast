"""
1D convolution functions.

Created: 2020-09-23
"""
import numpy as np
import scipy as sp
from scipy.sparse import diags as band


class Conv1D:

    @staticmethod
    def get_conv_matrix(signal, kernel):
        """
        Construct full convolution matrix (n+m-1) x n,
        where n is the signal length and m the kernel length.
        """
        n = signal.shape[0]
        m = kernel.shape[0]
        padding = np.zeros(n - 1)
        first_col = np.r_[kernel, padding]
        first_row = np.r_[kernel[0], padding]
        return sp.linalg.toeplitz(first_col, first_row)

    @staticmethod
    def freq_conv(signal, kernel):
        """1D convolution in the frequency domain"""
        n = signal.shape[0]
        m = kernel.shape[0]
        signal_freq = np.fft.fft(signal, n + m - 1)
        kernel_freq = np.fft.fft(kernel, n + m - 1)
        return np.fft.ifft(signal_freq * kernel_freq).real

    @staticmethod
    def freq_deconv(signal, kernel):
        """1D deconvolution in the frequency domain"""
        n = signal.shape[0]
        m = kernel.shape[0]
        signal_freq = np.fft.fft(signal, n + m - 1)
        kernel_freq = np.fft.fft(kernel, n + m - 1)
        return np.fft.ifft(signal_freq / kernel_freq).real


def soft_thresh(x, lam):
    """Perform soft-thresholding with threshold lambda."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)


def admm_deconvolution(y, kernel, lam, rho, n_iters=100, k=0):
    """
    Solves the following optimization problem via ADMM

    minimize  (1/2n) ||y - Cx||_2^2 + lam*||Dx||_1
       x

    where C is the discrete convolution matrix, and D the
    discrete differences matrix.
    """
    n = y.shape[0]
    m = kernel.shape[0]
    C = Conv1D.get_conv_matrix(y, kernel)[:n, ]
    D = band([-1, 1], [0, 1], shape=(n - 1, n)).toarray()
    D = np.diff(D, n=k, axis=0)
    # pre-calculations
    DtD = D.T @ D
    CtC = C.T @ C / n
    Cty = C.T @ y / n
    x_update_1 = np.linalg.inv(CtC + rho * DtD)

    x_k = None
    alpha_0 = np.zeros(n - k - 1)
    u_0 = np.zeros(n - k - 1)
    for t in range(n_iters):
        x_k = x_update_1 @ (Cty + rho * D.T @ (alpha_0 - u_0))
        alpha_k = soft_thresh(D @ x_k + u_0, lam / rho)
        u_k = u_0 + D @ x_k - alpha_k

        alpha_0 = alpha_k
        u_0 = u_k

    return x_k


def admm_deconvolution_v2(y, kernel, lam, rho, n_iters=100, k=0):
    """
    Solves the following optimization problem via ADMM

    minimize  (1/2n) ||iCy - x||_2^2 + lam*||Dx||_1
       x

    where iC is the inverted discrete convolution matrix, and
    D the discrete differences matrix.
    """
    n = y.shape[0]
    m = kernel.shape[0]
    C = Conv1D.get_conv_matrix(y, kernel)[:n, ]
    D = band([-1, 1], [0, 1], shape=(n - 1, n)).toarray()
    D = np.diff(D, n=k, axis=0)
    I = np.eye(n)

    # pre-calculations
    iC = np.linalg.inv(C)
    iCy = iC @ y / n
    DtD = D.T @ D
    x_update_1 = np.linalg.inv((I / n) + rho * DtD)

    x_k = None
    alpha_0 = np.zeros(n - k - 1)
    u_0 = np.zeros(n - k - 1)
    for t in range(n_iters):
        x_k = x_update_1 @ (iCy + rho * D.T @ (alpha_0 - u_0))
        alpha_k = soft_thresh(D @ x_k + u_0, lam / rho)
        u_k = u_0 + D @ x_k - alpha_k

        alpha_0 = alpha_k
        u_0 = u_k

    return x_k
