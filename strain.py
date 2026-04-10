#-*- coding: utf-8 -*-

import numpy as np
import sys


def compute_strain(
    vx,
    vy,
    dx=100,
    dy=-100,
    grad_method='numpy',
    inpaint=True,
    rotate=False,
    eps_min=1.0e-5,
    **kwargs
):
    """
    Compute stress and strain fields and return in dictionaries.

    Parameters
    ----------
    vx: (M, N) ndarray
        Input array for velocities in X-direction.
    vy: (M, N) ndarray
        Input array for velocities in Y-direction.
    dx: float, optional
        Spacing in X-direction.
    dy: float, optional
        Spacing in Y-direction.
    grad_method: str, optional
        Gradient method specifier in ('numpy', 'sgolay', 'robust_l2', 'robust_lp').
        Default: 'numpy'.
    inpaint: bool, optional
        Inpaint arrays prior to gradient computation. Default: True.
    rotate: bool, optional
        Rotate velocity gradients to along-flow direction. Default: False.
    eps_min: float, optional
        Minimum effective strain rate value for numerical stability. Default: 1e-5.
    **kwargs:
        Extra keyword arguments to pass to gradient computation. 

    Returns
    -------
    strain_dict: dict
        Dictionary of (M, N) arrays containing strain components 'e_xx', 'e_yy', 'e_xy',
        'dilatation', 'effective'.
    """
    # Cache image shape
    Ny, Nx = vx.shape

    # Compute velocity gradients
    L12, L11 = gradient(
        vx, spacing=(dy, dx), method=grad_method, inpaint=inpaint, **kwargs
    )
    L22, L21 = gradient(
        vy, spacing=(dy, dx), method=grad_method, inpaint=inpaint, **kwargs
    )

    # Compute components of strain-rate tensor
    D = np.empty((2, 2, vx.size))
    D[0, 0, :] = 0.5 * (L11 + L11).ravel()
    D[0, 1, :] = 0.5 * (L12 + L21).ravel()
    D[1, 0, :] = 0.5 * (L21 + L12).ravel()
    D[1, 1, :] = 0.5 * (L22 + L22).ravel()

    # Compute pixel-dependent rotation tensor if requested
    if rotate:
        R = np.empty((2, 2, vx.size))
        theta = np.arctan2(vy, vx).ravel()
        R[0, 0, :] = np.cos(theta)
        R[0, 1, :] = np.sin(theta)
        R[1, 0, :] = -np.sin(theta)
        R[1, 1, :] = np.cos(theta)

        # Apply rotation tensor
        D = np.einsum('ijm,kjm->ikm', D, R)
        D = np.einsum('ijm,jkm->ikm', R, D)

    # Cache elements of strain-rate tensor for easier viewing
    D11 = D[0, 0, :]
    D12 = D[0, 1, :]
    D21 = D[1, 0, :]
    D22 = D[1, 1, :]

    # Normal strain rates
    exx = D11.reshape(Ny, Nx)
    eyy = D22.reshape(Ny, Nx)
    exy = D12.reshape(Ny, Nx)

    # Shear- same result as
    # -> e_xy_max = |e_max - e_min|
    # -> e_xy_max = 2 * np.sqrt(0.25 * (e_x - e_y)**2 + e_xy**2)
    trace = (D11 + D22).reshape(Ny, Nx)
    det = (D11 * D22 - D12 * D21).reshape(Ny, Nx)
    shear = np.sqrt(0.25 * trace**2 - det)

    # Compute scalar quantities from stress tensors
    dilatation = (L11 + L22).reshape(Ny, Nx)
    effective_strain = np.sqrt(
        L11**2 + L22**2 + 0.25 * (L12 + L21)**2 + L11 * L22 + eps_min**2
    ).reshape(Ny, Nx)
    
    # Principal strain rates. Equivalent to
    # e_max/min = 0.5 * (e_x + e_y) +/- sqrt(0.25 * (e_x - e_y)**2 + e_xy**2)
    e_max = 0.5 * trace + 0.5 * shear
    e_min = 0.5 * trace - 0.5 * shear

    # Compute the angle of the first principal axis (orientation of e_max)
    principal_angle_rad = (0.5 * np.arctan2(2 * D12, D11 - D22)).reshape(Ny, Nx)
    principal_angle_deg = np.rad2deg(principal_angle_rad)

    # Store strain components in dictionary
    strain_dict = {
        'e_xx': exx,
        'e_yy': eyy,
        'e_xy': exy,
        'shear_magnitude': shear,
        'dilatation': dilatation,
        'effective': effective_strain,
        'e_max': e_max,
        'e_min': e_min,
        'principal_angle': principal_angle_deg,
    }

    # Return strain and stress dictionaries
    return strain_dict


def gradient(z, spacing=1.0, axis=None, remask=True, method='numpy',
             inpaint=False, **kwargs):
    """
    Calls either Numpy or Savitzky-Golay gradient computation routines.

    Parameters
    ----------
    z: Raster or array_like
        2-dimensional raster or array containing samples of a scalar function.
    spacing: float or tuple of floats, optional
        Spacing between f values along specified axes. If tuple provided, spacing corresponds
        to axes directions specified by axis. Default: 1.0.
    axis: None or int or tuple of ints, optional
        Axis or axes to compute gradient. If None, derivative computed along all
        dimensions. Default: 0.
    remask: bool, optional
        Apply NaN mask on gradients. Default: True.
    method: str, optional
        Method specifier in ('numpy', 'sgolay', 'robust'). Default: 'numpy'.
    inpaint: bool, optional
        Inpaint image prior to gradient computation (recommended for
        'sgolay' method). Default: False.
    **kwargs:
        Extra keyword arguments to pass to specific gradient computation.

    Returns
    -------
    s: ndarray or list of ndarray
        Set of ndarrays (or single ndarry for only one axis) with same shape as z
        corresponding to the derivatives of z with respect to each axis.
    """
    # Mask mask of NaNs
    nan_mask = np.isnan(z)
    have_nan = np.any(nan_mask)

    # For sgolay method, we need to inpaint if NaNs detected
    if inpaint and have_nan:
        z_inp = inpaint_spring(z, mask=nan_mask)
    else:
        z_inp = z

    # Compute gradient with numpy
    if method == 'numpy':
        if isinstance(spacing, (tuple, list)):
            s = np.gradient(z_inp, spacing[0], spacing[1], axis=(0, 1), edge_order=2)
        else:
            s = np.gradient(z_inp, spacing, axis=axis, edge_order=2)

    # With Savtizky-Golay
    elif method == 'sgolay':
        s = sgolay_gradient(z_inp, spacing=spacing, axis=axis, **kwargs)

    # With robust polynomial
    elif method in ('robust_l2', 'robust_lp'):
        zs, z_dy, z_dx = robust_gradient(z_inp, spacing=spacing, lsq_method=method, **kwargs)
        s = (z_dy, z_dx)
        if axis is not None and isinstance(axis, int):
            s = s[axis]

    else:
        raise ValueError('Unsupported gradient method.')

    # Re-apply mask
    if remask and have_nan:
        if isinstance(s, (tuple, list)):
            for arr in s:
                arr[nan_mask] = np.nan
        else:
            s[nan_mask] = np.nan

    return s


def sgolay_gradient(z, spacing=1.0, axis=None, window_size=3, order=4, pad='reflect'):
    """
    Wrapper around Savitzky-Golay code to compute window size in pixels and call _sgolay2d
    with correct arguments.

    Parameters
    ----------
    z: array_like
        2-dimensional array containing samples of a scalar function.
    spacing: float or tuple of floats, optional
        Spacing between f values along specified axes. If tuple provided, spacing is
        specified as (dy, dx) and derivative computed along both dimensions.
        Default is unitary spacing.
    axis: None or int, optional
        Axis along which to compute gradients. If None, gradient computed
        along both dimensions. Default: None.
    window_size: scalar or tuple of scalars, optional
        Window size in units of specified spacing. If tuple provided, window size is
        specified as (win_y, win_x). Default: 3.
    order: int, optional
        Polynomial order. Default: 4.
    pad: str, optional
        Option for np.pad kwarg 'mode'. Default: 'reflect'.

    Returns
    -------
    gradient: a
        Array or tuple of array corresponding to gradients. If both axes directions
        are specified, returns (dz/dy, dz/dx).
    """
    # Compute derivatives in both directions
    if axis is None or isinstance(spacing, (tuple, list)):

        # Compute window sizes
        wy, wx = compute_windows(window_size, spacing)

        # Unpack spacing
        dy, dx = spacing
        
        # Call Savitzky-Golay twice in order to use different window sizes
        sy = _sgolay2d(z, wy, order=order, derivative='col', pad=pad)
        sx = _sgolay2d(z, wx, order=order, derivative='row', pad=pad)

        # Scale by spacing and return
        return sy / dy, sx / dx

    # Or derivative in a single direction
    else:

        assert axis is not None, 'Must specify axis direction.'

        # Compute window size
        w = int(np.ceil(abs(window_size / spacing)))
        if w % 2 == 0:
            w += 1

        # Call Savitzky-Golay
        if axis == 0:
            s = _sgolay2d(z, w, order=order, derivative='col', pad=pad)
        elif axis == 1:
            s = _sgolay2d(z, w, order=order, derivative='row', pad=pad)
        else:
            raise ValueError('Axis must be 0 or 1.')

        # Scale by spacing and return
        return s / spacing


def _sgolay2d(z, window_size, order, derivative=None, pad='reflect'):
    """
    Max Filter, January 2021.

    Original lower-level code from Scipy cookbook, with modifications to
    padding.
    """
    from scipy.signal import fftconvolve

    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial. 
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ... 
    # this line gives a list of two item tuple. Each tuple contains 
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    Z = np.pad(z, half_size, mode=pad)

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        Zf = fftconvolve(Z, m, mode='valid')
        return Zf
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        Zc = fftconvolve(Z, -c, mode='valid')
        return Zc
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        Zr = fftconvolve(Z, -r, mode='valid')
        return Zr
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        Zc = fftconvolve(Z, -c, mode='valid')
        Zr = fftconvolve(Z, -r, mode='valid')
        return Zr, Zc


def compute_windows(window_size, spacing):
    """
    Convenience function to compute window sizes in pixels given spacing of
    pixels in physical coordinates.

    Parameters
    ----------
    spacing: scalar or tuple of scalars
        Spacing between pixels along axis. If tuple, each element specifies
        spacing along different axes.
    window_size: scalar or tuple of scalars
        Window size in units of specified spacing. If tuple provided, window size is
        specified as (win_y, win_x).

    Returns
    -------
    w: tuple of scalars
        Odd-number window sizes in both axes directions in number of pixels.
    """
    # Unpack spacing
    if isinstance(spacing, (tuple, list)):
        assert len(spacing) == 2, 'Spacing must be 2-element tuple.'
        dy, dx = spacing
    else:
        dy = dx = spacing

    # Compute window sizes
    if isinstance(window_size, (tuple, list)):
        assert len(window_size) == 2, 'Window size must be 2-element tuple.'
        wy, wx = window_size
    else:
        wy = wx = window_size

    # Array of windows
    if isinstance(wy, np.ndarray):
        wy = np.ceil(np.abs(wy / dy)).astype(int)
        wx = np.ceil(np.abs(wx / dx)).astype(int)
        wy[(wy % 2) == 0] += 1
        wx[(wx % 2) == 0] += 1

    # Or scalar windows
    else:
        wy, wx = int(np.ceil(abs(wy / dy))), int(np.ceil(abs(wx / dx)))
        wy = _make_odd(wy)
        wx = _make_odd(wx)

    return wy, wx


def _make_odd(w):
    """
    Convenience function to ensure a number is odd.
    """
    if w % 2 == 0:
        w += 1
    return w

# Lambda for getting order-dependent polynomial exponents
_compute_exps = lambda order: [(k-n, n) for k in range(order + 1) for n in range(k + 1)]


def inpaint_spring(ain, mask):
    '''Returns the inpainted matrix using the spring metaphor.
       All NaN values in the matrix are filled in.
       
       Based on the original inpaintnans package by John D'Errico.
       http://www.mathworks.com/matlabcentral/fileexchange/4551-inpaintnans'''
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla

    dims = ain.shape
    bout = ain.copy()
    nnn = dims[0]
    mmm = dims[1]
    nnmm = nnn * mmm

    [iii, jjj] = np.where(mask == False)
    [iin, jjn] = np.where(mask == True)
    nnan = len(iin)    #Number of nan.

    if nnan == 0:
        return bout

    hv_springs = np.zeros((4 * nnan, 2), dtype=int)
    cnt = 0
    for kkk in range(nnan):
        ypos = iin[kkk]
        xpos = jjn[kkk]
        indc = ypos * mmm + xpos
        if(ypos > 0):
            hv_springs[cnt, :]   = [indc - mmm, indc]   #Top
            cnt = cnt + 1

        if(ypos < (nnn - 1)):
            hv_springs[cnt, :] = [indc, indc + mmm]  #Bottom
            cnt = cnt + 1

        if(xpos>0):
            hv_springs[cnt, :] = [indc - 1, indc]  #Left
            cnt = cnt + 1

        if(xpos < (mmm - 1)):
            hv_springs[cnt, :] = [indc, indc + 1]  #Right
            cnt = cnt + 1

    hv_springs = hv_springs[0:cnt, :]

    tempb = _unique_rows(hv_springs)
    cnt = tempb.shape[0]

    alarge = sp.csc_matrix((np.ones(cnt), (np.arange(cnt), tempb[:, 0])),
            shape=(cnt, nnmm))
    alarge = alarge + sp.csc_matrix((-np.ones(cnt), (np.arange(cnt)
        , tempb[:, 1])), shape=(cnt, nnmm))

    indk = iii * mmm + jjj
    indu = iin * mmm + jjn
    dkk  = -ain[iii, jjj]
    del iii
    del jjj

    aknown = sp.csc_matrix(alarge[:, indk])
    rhs = sp.csc_matrix.dot(aknown, dkk)
    del aknown
    del dkk
    anan = sp.csc_matrix(alarge[:, indu])
    dku = sla.lsqr(anan, rhs)
    bout[iin, jjn] = dku[0]
    return bout


def _unique_rows(scenes):
    '''Unique rows utility similar to matlab.'''
    uscenes = np.unique(scenes.view([('',scenes.dtype)]*scenes.shape[1])).view(scenes.dtype).reshape(-1,scenes.shape[1])
    return uscenes



# end of file
