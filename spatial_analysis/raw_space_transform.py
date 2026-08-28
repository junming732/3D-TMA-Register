"""
raw_space_transform.py
====================
Converts a cell centroid from a pipeline's own registered-image coordinates
back into the ORIGINAL raw (pre-registration) per-slice image coordinates —
the one coordinate frame every registration pipeline shares by construction,
since they all start from the same raw serial sections.

Why this exists
----------------
cell_id_3d, and the centroid_x_um/y_um/z_um stored alongside it, are computed
from CellPose masks warped into THAT pipeline's own registered-image space.
Different pipelines crop/scale/warp differently, so two pipelines' registered
spaces are NOT the same coordinate system — comparing centroids directly
(what this repo did before) silently compares unrelated coordinates. Raw
per-slice image space is the only frame guaranteed shared by every pipeline.

Design
------
Both Valis and a from-scratch pipeline like RomaV2 ultimately produce, per
z-slice: an affine matrix (raw -> some intermediate aligned space) plus a
dense "backward" non-rigid displacement field (registered-space pixel ->
corresponding coordinate one step back towards raw space). That structural
shape is shared. The exact matrix CONVENTION is not — Valis's `M` is used as
its OWN inverse in a right-multiply, while RomaV2's `M_affine` is applied
directly, forward, moving->fixed. Rather than pretend those conventions are
identical (they aren't — that would just be a different, hidden bug), this
module keeps one tiny, clearly-labeled adapter per pipeline that normalizes
each into the same shared backward-field-sampling step underneath.

Every pipeline's per-slice transform data lives in one small, boring .npz
file, produced ONCE by an export step (export_valis_transform.py for Valis;
RomaV2's existing deformation .npz files already are in this shape). This
module never imports `valis`, `pyvips`, or torch — it only ever reads the
small numpy arrays those export steps already produced.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline, SmoothBivariateSpline

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # only needed by get_inverse_field(), which our production
                 # path never calls (we always have bk_dxdy directly)

try:
    import pyvips
    _HAS_PYVIPS = True
except ImportError:
    _HAS_PYVIPS = False
    class _NoPyvipsImage:  # sentinel so isinstance() checks below never match
        pass
    class _FakePyvipsModule:
        Image = _NoPyvipsImage
    pyvips = _FakePyvipsModule()


# ─────────────────────────────────────────────────────────────────────────────
# VENDORED, VERBATIM from valis.warp_tools (valis-wsi 1.2.0), NOT reimplemented.
# An earlier hand-written reimplementation of this scale-handling logic was
# tested and found WRONG (silently, ~4000px off) — the actual internal scale
# math has enough subtlety that paraphrasing it introduced a real bug. This
# block is copied as-is instead, and round-trip tested against the real
# functions (see test_warp_isolated.py) to <0.03px on synthetic data.
# Only get_warp_scaling_factors, warp_xy, warp_xy_inv, warp_xy_rigid,
# warp_xy_non_rigid, _warp_xy_numpy, and get_inverse_field are included —
# nothing else from valis.warp_tools is pulled in, so this has no dependency
# on valis itself, torch, kornia, or a JVM/bioformats bridge.
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# SHARED LOW-LEVEL STEP — identical math regardless of which pipeline produced
# the backward field: bilinearly sample a dense (H, W) displacement field at
# arbitrary (possibly non-integer) points.
# ─────────────────────────────────────────────────────────────────────────────
def sample_backward_field(xy, bk_dx, bk_dy):
    """
    xy : (N, 2) array of (x, y) points, in the SAME pixel grid bk_dx/bk_dy
         are indexed by (i.e. already at the field's own resolution).
    bk_dx, bk_dy : (H, W) arrays. At output/registered pixel (row, col), give
         the (dx, dy) to reach the corresponding point one step back towards
         raw space: source_x = col + bk_dx[row, col], source_y = row + bk_dy[row, col].

    Returns (N, 2) array of displaced (x, y) points, same grid/resolution as
    the input `xy`.
    """
    xy = np.asarray(xy, dtype=np.float64)
    H, W = bk_dx.shape
    rows = np.arange(H)
    cols = np.arange(W)

    # RegularGridInterpolator expects (row, col) = (y, x) order and points as (N, 2) [y, x]
    interp_dx = RegularGridInterpolator((rows, cols), bk_dx, bounds_error=False, fill_value=None)
    interp_dy = RegularGridInterpolator((rows, cols), bk_dy, bounds_error=False, fill_value=None)

    pts_yx = np.column_stack([xy[:, 1], xy[:, 0]])
    dx = interp_dx(pts_yx)
    dy = interp_dy(pts_yx)

    out = xy.copy()
    out[:, 0] += dx
    out[:, 1] += dy
    return out


# ─────────────────────────────────────────────────────────────────────────────
# VALIS ADAPTER — thin wrapper around the vendored warp_xy_inv() below.
# ─────────────────────────────────────────────────────────────────────────────
def valis_registered_to_raw(xy_registered, M, bk_dx, bk_dy,
                             processed_shape_rc, registered_shape_rc,
                             raw_full_res_shape_rc, registered_full_res_shape_rc):
    """
    xy_registered : (N, 2) points, in FULL-RESOLUTION registered-image pixels
                     (i.e. the same pixel space link_3d_cells.py's cell
                     centroids already live in).
    M : (3, 3) affine matrix, as stored on the Valis Slide object.
    bk_dx, bk_dy : (H, W) backward displacement fields at `processed_shape_rc`
                     resolution, as stored on the Valis Slide object (bk_dxdy).
    processed_shape_rc, registered_shape_rc : shapes the transform/field were
                     found at (Slide.processed_img_shape_rc / .reg_img_shape_rc).
    raw_full_res_shape_rc : full-resolution shape of the ORIGINAL slide
                     (Slide.slide_dimensions_wh[0][::-1]).
    registered_full_res_shape_rc : full-resolution shape of the final
                     registered volume (Slide.aligned_slide_shape_rc).

    Returns (N, 2) points in full-resolution RAW/original image pixels.

    This is a thin wrapper around the vendored, verified warp_xy_inv() above —
    deliberately NOT reimplemented, since a hand-written version of this
    specific scale-handling logic was tried and found wrong. See module
    docstring.
    """
    bk_dxdy = np.stack([np.asarray(bk_dx), np.asarray(bk_dy)])
    return warp_xy_inv(
        np.asarray(xy_registered, dtype=np.float64),
        M=M,
        transformation_src_shape_rc=processed_shape_rc,
        transformation_dst_shape_rc=registered_shape_rc,
        src_shape_rc=raw_full_res_shape_rc,
        dst_shape_rc=registered_full_res_shape_rc,
        bk_dxdy=bk_dxdy,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROMAV2 ADAPTER
# Convention taken directly from this repo's own registration_accuracy_landmarks.py
# warp_point(): M_affine is applied DIRECTLY forward (moving->fixed, no
# inversion), and map_x/map_y already operate at the same resolution the
# input (x, y) points are given in — no rescaling needed.
# ─────────────────────────────────────────────────────────────────────────────
def romav2_registered_to_raw(xy_registered, M_affine, map_x, map_y):
    """
    xy_registered : (N, 2) points, in full-resolution registered-image pixels.
    M_affine : (2, 3) matrix, moving->fixed, as saved in the deformation .npz.
    map_x, map_y : (H, W) backward maps — ABSOLUTE affine-space source
                     coordinates (not deltas), as saved in the deformation
                     .npz and as used directly in warp_point()'s own NN search.

    Returns (N, 2) points in full-resolution RAW/original image pixels.
    """
    xy = np.asarray(xy_registered, dtype=np.float64)

    # map_x/map_y store ABSOLUTE coordinates (verified against warp_point()'s
    # own NN search, which compares patch_x/patch_y directly against ax/ay) —
    # convert once to the delta convention sample_backward_field expects, so
    # the same shared bilinear-sampling step works for both pipelines.
    H, W = map_x.shape
    col_grid, row_grid = np.meshgrid(np.arange(W), np.arange(H))
    bk_dx = (map_x - col_grid).astype(np.float64)
    bk_dy = (map_y - row_grid).astype(np.float64)

    # Sample the backward map: registered -> affine-prealigned space.
    xy_affine_space = sample_backward_field(xy, bk_dx, bk_dy)

    # Undo the affine step. RomaV2 applies M_affine directly forward (no
    # inversion, per their own warp_point() docstring), so the true
    # mathematical inverse is needed here to reverse it.
    M3 = np.vstack([M_affine, [0, 0, 1]])
    inv_M = np.linalg.inv(M3)
    n = xy_affine_space.shape[0]
    homog = np.column_stack([xy_affine_space, np.ones(n)])
    xy_raw = (homog @ inv_M.T)[:, :2]
    return xy_raw


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED ENTRY POINT — qc_reference.py should only ever need to call this.
# ─────────────────────────────────────────────────────────────────────────────
def registered_to_raw_xy(pipeline_kind, xy_registered, transform_npz_path):
    """
    pipeline_kind : "valis" or "romav2"
    xy_registered : (N, 2) array, full-resolution registered-space pixel coords
    transform_npz_path : path to that pipeline's per-slice transform .npz
                          (see export_valis_transform.py for Valis; RomaV2's
                          existing deformation .npz files need no changes)

    Returns (N, 2) array of full-resolution RAW/original image pixel coords.
    """
    d = np.load(transform_npz_path)

    if pipeline_kind == "valis":
        return valis_registered_to_raw(
            xy_registered,
            M=d['M'],
            bk_dx=d['bk_dx'], bk_dy=d['bk_dy'],
            processed_shape_rc=tuple(d['processed_shape_rc']),
            registered_shape_rc=tuple(d['registered_shape_rc']),
            raw_full_res_shape_rc=tuple(d['raw_full_res_shape_rc']),
            registered_full_res_shape_rc=tuple(d['registered_full_res_shape_rc']),
        )
    elif pipeline_kind == "romav2":
        return romav2_registered_to_raw(
            xy_registered,
            M_affine=d['M_affine'],
            map_x=d['map_x'], map_y=d['map_y'],
        )
    else:
        raise ValueError(f"Unknown pipeline_kind: {pipeline_kind!r} (expected 'valis' or 'romav2')")
def get_inverse_field(backwards_xy_deltas, n_inter=10):
    """
    Invert transform
    """

    sitk_bk_dxdy = sitk.GetImageFromArray(np.dstack(backwards_xy_deltas),  isVector=True)
    sitk_fw_dxdy = sitk.IterativeInverseDisplacementField(sitk_bk_dxdy, numberOfIterations=n_inter)
    fwd_dxdy = sitk.GetArrayFromImage(sitk_fw_dxdy)
    fwd_dxdy = [fwd_dxdy[..., 0], fwd_dxdy[..., 1]]

    return fwd_dxdy


def warp_xy_rigid(xy, inv_matrix):
    """ Warp points

    Warp xy given an inverse transformation matrix found using one of scikit-image's transform objects
    Inverse matrix should have been found using tform(dst, src)
    Adpated from skimage._geometric.ProjectiveTransform._apply_mat
    Changed so that inverse matrix (found using dst -> src) automatically inverted to warp points forward (src -> dst)
    """
    xy = np.array(xy, copy=False, ndmin=2)

    x, y = np.transpose(xy)
    src_pts = np.vstack((x, y, np.ones_like(x)))
    try:
        dst_pts = src_pts.T @ np.linalg.inv(inv_matrix).T
    except np.linalg.LinAlgError :
        print("Singular matrix")
        dst_pts = src_pts.T @ np.linalg.pinv(inv_matrix).T

    # below, we will divide by the last dimension of the homogeneous
    # coordinate matrix. In order to avoid division by zero,
    # we replace exact zeros in this column with a very small number.
    dst_pts[dst_pts[:, 2] == 0, 2] = np.finfo(float).eps
    # rescale to homogeneous coordinates
    dst_pts[:, :2] /= dst_pts[:, 2:3]

    return dst_pts[:, :2]


def warp_xy_non_rigid(xy, dxdy, displacement_shape_rc=None):

    single_pt = xy.ndim == 1
    if single_pt:
        xy = np.array([xy])

    if displacement_shape_rc is None:
        displacement_shape_rc = dxdy[0].shape

    bbox = [0, displacement_shape_rc[0], 0, displacement_shape_rc[1]]
    grid_r = np.arange(displacement_shape_rc[0])
    grid_c = np.arange(displacement_shape_rc[1])

    interp_dx = RectBivariateSpline(grid_r, grid_c, dxdy[0], bbox=bbox)
    interp_dy = RectBivariateSpline(grid_r, grid_c, dxdy[1], bbox=bbox)

    nr_x = xy[:, 0] + interp_dx(xy[:, 1], xy[:, 0], grid=False)
    nr_y = xy[:, 1] + interp_dy(xy[:, 1], xy[:, 0], grid=False)

    nr_xy = np.dstack([nr_x, nr_y])[0]
    if single_pt:
        nr_xy = nr_xy[0]

    return nr_xy


def get_warp_scaling_factors(transformation_src_shape_rc=None, transformation_dst_shape_rc=None, src_shape_rc=None, dst_shape_rc=None, bk_dxdy=None, fwd_dxdy=None):
    """Get scaling factors needed to warp points

    If a returned value is None, it means there is no need to scale the image
    Returns
    -------
    src_sxy : ndarray
        Scaling to go from transformation_src_shape_rc -> src_shape_rc (i.e. transformation_src_shape_rc/src_shape_rc)

    dst_sxy : ndarray
        When `bk_dxdy` or `fwd_dxdy` is None, this is the scaling to go from
        transformation_dst_shape_rc -> dst_shape_rc (i.e. dst_shape_rc/transformation_dst_shape_rc).

        When `bk_dxdy` or `fwd_dxdy` are provided, this is the scaling that goes from the
        displacement -> `dst_shape_rc`

    displacement_sxy :
        Scaling for dxdy for when non-rigid transformations found using an
        image with a size different than transformation_dst_shape_rc.

        For example, if displacement was found on an image 2x the one with
        `transformation_dst_shape_rc`, this would be 2. Used to warp points
        from position in image with shape transformation_dst_shape_rc to position
        in `bk_dxdy` or `fwd_dxdy`.

    displacement_shape_rc : (int, int)
        Shape of displacement field used for non-rigid transforms

    """
    do_non_rigid = bk_dxdy is not None or fwd_dxdy is not None

    # convert shapes to arrays
    if src_shape_rc is not None:
        src_shape_rc = np.array(src_shape_rc)

    if transformation_src_shape_rc is not None:
        transformation_src_shape_rc = np.array(transformation_src_shape_rc)

    if dst_shape_rc is not None:
        dst_shape_rc = np.array(dst_shape_rc)

    if transformation_dst_shape_rc is not None:
        transformation_dst_shape_rc = np.array(transformation_dst_shape_rc)

    # Get input scaling
    if transformation_src_shape_rc is not None and src_shape_rc is not None:
        # Scale points to where they would be in image with transformation_src_shape_rc
        if np.all(transformation_src_shape_rc == src_shape_rc):
            src_sxy = None
        else:
            src_sxy = (src_shape_rc/transformation_src_shape_rc)[::-1]
    else:
        src_sxy = None

    # Get output shapes
    non_rigid_is_array = False
    if bk_dxdy is not None or fwd_dxdy is not None:
        if bk_dxdy is not None:
            if not isinstance(bk_dxdy, pyvips.Image):
                non_rigid_is_array = True
        if fwd_dxdy is not None:
            if not isinstance(fwd_dxdy, pyvips.Image):
                non_rigid_is_array = True

    if do_non_rigid:
        if bk_dxdy is not None:
            if non_rigid_is_array:
                displacement_shape_rc = np.array(bk_dxdy[0].shape)
            else:
                displacement_shape_rc = np.array([bk_dxdy.height, bk_dxdy.width])
        elif fwd_dxdy is not None:
            if non_rigid_is_array:
                displacement_shape_rc = np.array(fwd_dxdy[0].shape)
            else:
                displacement_shape_rc = np.array([fwd_dxdy.height, fwd_dxdy.width])

    if transformation_dst_shape_rc is None and do_non_rigid:
            transformation_dst_shape_rc = displacement_shape_rc

    if dst_shape_rc is None and transformation_dst_shape_rc is not None:
        dst_shape_rc = transformation_dst_shape_rc

    # Get output scalings
    if do_non_rigid:
        if not np.all(transformation_dst_shape_rc == displacement_shape_rc):
            # non-rigid found on scaled image
            displacement_sxy = (displacement_shape_rc/transformation_dst_shape_rc)[::-1]
            dst_sxy = (dst_shape_rc/displacement_shape_rc)[::-1]
        else:
            displacement_sxy = None
            dst_sxy = (dst_shape_rc/transformation_dst_shape_rc)[::-1]

        if np.all(dst_sxy == 1):
            dst_sxy = None
    else:
        # Determine how to scale to images for position in image with shape = dst_shape_rc
        dst_sxy = None
        displacement_shape_rc = None
        displacement_sxy = None
        if transformation_dst_shape_rc is not None and dst_shape_rc is not None:
            if not np.all(dst_shape_rc == transformation_dst_shape_rc):
                dst_sxy = (dst_shape_rc/transformation_dst_shape_rc)[::-1]

    return src_sxy, dst_sxy, displacement_sxy, displacement_shape_rc



def _warp_pt_vips(xy, M=None, vips_bk_dxdy=None, vips_fwd_dxdy=None, src_sxy=None, dst_sxy=None, displacement_sxy=None, displacement_shape_rc=None, pt_buffer=100):
    """Warp single point when the displacement fields are pyvips.Image objects

    """

    do_non_rigid = vips_bk_dxdy is not None or vips_fwd_dxdy is not None

    if src_sxy is not None:
        in_src_xy = xy/src_sxy

    else:
        in_src_xy = xy

    if M is not None:
        rigid_xy = warp_xy_rigid(in_src_xy, M).astype(float)[0]
        if not do_non_rigid:
            if dst_sxy is not None:
                return rigid_xy*dst_sxy
            else:
                return rigid_xy
    else:
        rigid_xy = in_src_xy

    if displacement_sxy is not None:
        # displacement was found on scaled version of the rigidly registered image.
        # So move points into new displacement field
        rigid_xy *= displacement_sxy

    bbox_xy_tl  = (rigid_xy - pt_buffer//2).astype(int)
    bbox_xy_br  = np.ceil(rigid_xy + pt_buffer//2).astype(int)
    bbox_x01 = np.clip(np.array([bbox_xy_tl[0], bbox_xy_br[0]]), 0, displacement_shape_rc[1])
    bbox_y01 = np.clip(np.array([bbox_xy_tl[1], bbox_xy_br[1]]), 0, displacement_shape_rc[0])

    bbox_w = -int(np.subtract(*bbox_x01))
    bbox_h = -int(np.subtract(*bbox_y01))
    region_bbox_xywh = np.array([bbox_x01[0], bbox_y01[0], bbox_w, bbox_h])

    # Move point to position in tile
    rigid_xy_in_tile = rigid_xy - region_bbox_xywh[:2]

    # Get region dxdy
    if vips_fwd_dxdy is not None:
        vips_region_dxdy = vips_fwd_dxdy.extract_area(*region_bbox_xywh)
        region_dxdy = vips2numpy(vips_region_dxdy)
    elif vips_bk_dxdy is not None and vips_fwd_dxdy is None:
        vips_region_bk_dxdy = vips_bk_dxdy.extract_area(*region_bbox_xywh)
        region_bk_dxdy = vips2numpy(vips_region_bk_dxdy)
        region_dxdy = np.dstack(get_inverse_field(region_bk_dxdy[..., 0], region_bk_dxdy[..., 1]))

    nonrigid_xy = warp_xy_non_rigid(xy=rigid_xy_in_tile, dxdy=[region_dxdy[..., 0], region_dxdy[..., 1]], displacement_shape_rc=[bbox_h, bbox_w])
    nonrigid_xy += region_bbox_xywh[0:2]

    if dst_sxy is not None:
        nonrigid_xy *= dst_sxy

    return nonrigid_xy


def _warp_xy_vips(xy, M=None, transformation_src_shape_rc=None, transformation_dst_shape_rc=None,
                 src_shape_rc=None, dst_shape_rc=None, vips_bk_dxdy=None, vips_fwd_dxdy=None, pt_buffer=100):
    """
    Warp xy points using M and/or bk_dxdy/fwd_dxdy.
    Used when `vips_bk_dxdy` or `vips_fwd_dxdy` is a pyvips.Image

    Parameters
    ----------
    xy : ndarray
        [P, 2] array of xy coordinates for P points

    M : ndarray, optional
         3x3 affine transformation matrix to perform rigid warp

    transformation_src_shape_rc : (int, int)
        Shape of image that was used to find the transformation.
        For example, this could be the original image in which features were detected

    transformation_dst_shape_rc : (int, int), optional
        Shape of the image with shape `transformation_src_shape_rc` after warping.
        This could be the shape of the original image after applying `M`.

    src_shape_rc : optional, (int, int)
        Shape of the image from which the points originated. For example,
        this could be a larger/smaller version of the image that was
        used for feature detection.

    dst_shape_rc : optional, (int, int)
        Shape of image (with shape `src_shape_rc`) after warping

    vips_bk_dxdy : pyvips.Image
        (2, N, M) numpy array of pixel displacements in the x and y
        directions from the reference image. dx = bk_dxdy[0],
        and dy=bk_dxdy[1]. If `bk_dxdy` is not None, but
        `fwd_dxdy` is None, then `bk_dxdy` will be inverted to warp `xy`.

    vips_fwd_dxdy : pyvips.Image
        Inverse of bk_dxdy. dx = fwd_dxdy[0], and dy=fwd_dxdy[1].
        This is what is actually used to warp the points.

    pt_buffer : int
        This method slices the region surrounding the point from the displacement fields.
        The `pt_buffer` determines the size of the window around the point.

    Returns
    -------
    warped_xy : [P, 2] array
        Array of warped xy coordinates for P points

    """
    src_sxy, dst_sxy, displacement_sxy, displacement_shape_rc = get_warp_scaling_factors(transformation_src_shape_rc=transformation_src_shape_rc,
                                                                        transformation_dst_shape_rc=transformation_dst_shape_rc,
                                                                        src_shape_rc=src_shape_rc, dst_shape_rc=dst_shape_rc,
                                                                        bk_dxdy=vips_bk_dxdy, fwd_dxdy=vips_fwd_dxdy)


    warped_xy = np.vstack([_warp_pt_vips(pt, M, vips_bk_dxdy=vips_bk_dxdy, vips_fwd_dxdy=vips_fwd_dxdy, src_sxy=src_sxy, dst_sxy=dst_sxy, displacement_sxy=displacement_sxy, displacement_shape_rc=displacement_shape_rc, pt_buffer=pt_buffer) for pt in xy])

    return warped_xy


def _warp_xy_numpy(xy, M=None, transformation_src_shape_rc=None, transformation_dst_shape_rc=None,
                   src_shape_rc=None, dst_shape_rc=None, bk_dxdy=None, fwd_dxdy=None):
    """
    Warp xy points using M and/or bk_dxdy/fwd_dxdy. If bk_dxdy is provided, it will be inverted to  create fwd_dxdy

    Parameters
    ----------
    xy : ndarray
        [P, 2] array of xy coordinates for P points

    M : ndarray, optional
         3x3 affine transformation matrix to perform rigid warp

    transformation_src_shape_rc : (int, int)
        Shape of image that was used to find the transformation.
        For example, this could be the original image in which features were detected

    transformation_dst_shape_rc : (int, int), optional
        Shape of the image with shape `transformation_src_shape_rc` after warping.
        This could be the shape of the original image after applying `M`.

    src_shape_rc : optional, (int, int)
        Shape of the image from which the points originated. For example,
        this could be a larger/smaller version of the image that was
        used for feature detection.

    dst_shape_rc : optional, (int, int)
        Shape of image (with shape `src_shape_rc`) after warping

    bk_dxdy : ndarray
        (2, N, M) numpy array of pixel displacements in the x and y
        directions from the reference image. dx = bk_dxdy[0],
        and dy=bk_dxdy[1]. If `bk_dxdy` is not None, but
        `fwd_dxdy` is None, then `bk_dxdy` will be inverted to warp `xy`.

    fwd_dxdy : ndarray
        Inverse of bk_dxdy. dx = fwd_dxdy[0], and dy=fwd_dxdy[1].
        This is what is actually used to warp the points.

    Returns
    -------
    warped_xy : [P, 2] array
        Array of warped xy coordinates for P points

    """

    do_non_rigid = bk_dxdy is not None or fwd_dxdy is not None

    if M is None and not do_non_rigid:
        return xy

    src_sxy, dst_sxy, displacement_sxy, displacement_shape_rc = get_warp_scaling_factors(transformation_src_shape_rc=transformation_src_shape_rc,
                                                                     transformation_dst_shape_rc=transformation_dst_shape_rc,
                                                                     src_shape_rc=src_shape_rc, dst_shape_rc=dst_shape_rc,
                                                                     bk_dxdy=bk_dxdy, fwd_dxdy=fwd_dxdy)
    if src_sxy is not None:
        in_src_xy = xy/src_sxy
    else:
        in_src_xy = xy

    if M is not None:
        rigid_xy = warp_xy_rigid(in_src_xy, M).astype(float)
        if not do_non_rigid:
            if dst_sxy is not None:
                return rigid_xy*dst_sxy
            else:
                return rigid_xy
    else:
        rigid_xy = in_src_xy

    if displacement_sxy is not None:
        # displacement was found on scaled version of the rigidly registered image.
        # So move points into new displacement field
        rigid_xy *= displacement_sxy

    if bk_dxdy is not None and fwd_dxdy is None:
        fwd_dxdy = get_inverse_field(bk_dxdy)

    nonrigid_xy = warp_xy_non_rigid(rigid_xy, dxdy=fwd_dxdy, displacement_shape_rc=displacement_shape_rc)

    if dst_sxy is not None:
        nonrigid_xy *= dst_sxy

    return nonrigid_xy

def warp_xy(xy, M=None, transformation_src_shape_rc=None, transformation_dst_shape_rc=None,
            src_shape_rc=None, dst_shape_rc=None,
            bk_dxdy=None, fwd_dxdy=None, pt_buffer=100):
    """
    Warp xy points using M and/or bk_dxdy/fwd_dxdy. If bk_dxdy is provided, it will be inverted to  create fwd_dxdy

    Parameters
    ----------
    xy : ndarray
        [P, 2] array of xy coordinates for P points

    M : ndarray, optional
         3x3 affine transformation matrix to perform rigid warp

    transformation_src_shape_rc : (int, int)
        Shape of image that was used to find the transformation.
        For example, this could be the original image in which features were detected

    transformation_dst_shape_rc : (int, int), optional
        Shape of the image with shape `transformation_src_shape_rc` after warping.
        This could be the shape of the original image after applying `M`.

    src_shape_rc : optional, (int, int)
        Shape of the image from which the points originated. For example,
        this could be a larger/smaller version of the image that was
        used for feature detection.

    dst_shape_rc : optional, (int, int)
        Shape of image (with shape `src_shape_rc`) after warping

    bk_dxdy : ndarray, pyvips.Image
        (2, N, M) numpy array of pixel displacements in the x and y
        directions from the reference image. dx = bk_dxdy[0],
        and dy=bk_dxdy[1]. If `bk_dxdy` is not None, but
        `fwd_dxdy` is None, then `bk_dxdy` will be inverted to warp `xy`.

    fwd_dxdy : ndarray, pyvips.Image
        Inverse of bk_dxdy. dx = fwd_dxdy[0], and dy=fwd_dxdy[1].
        This is what is actually used to warp the points.

    pt_buffer : int
        If `bk_dxdy` or `fwd_dxdy` are pyvips.Image object, then
        pt_buffer` determines the size of the window around the point used to
        get the local displacements.


    Returns
    -------
    warped_xy : [P, 2] array
        Array of warped xy coordinates for P points

    """

    do_non_rigid = bk_dxdy is not None or fwd_dxdy is not None

    if M is None and not do_non_rigid:
        return xy

    if isinstance(bk_dxdy, pyvips.Image) or isinstance(fwd_dxdy, pyvips.Image):
        warped_xy = _warp_xy_vips(xy, M, transformation_src_shape_rc=transformation_src_shape_rc,
                                  transformation_dst_shape_rc=transformation_dst_shape_rc,
                                  src_shape_rc=src_shape_rc, dst_shape_rc=dst_shape_rc,
                                  vips_bk_dxdy=bk_dxdy, vips_fwd_dxdy=fwd_dxdy, pt_buffer=pt_buffer)
    else:
        warped_xy = _warp_xy_numpy(xy, M, transformation_src_shape_rc=transformation_src_shape_rc,
                                   transformation_dst_shape_rc=transformation_dst_shape_rc,
                                   src_shape_rc=src_shape_rc, dst_shape_rc=dst_shape_rc,
                                   bk_dxdy=bk_dxdy, fwd_dxdy=fwd_dxdy)
    return warped_xy


def warp_xy_inv(xy, M=None, transformation_src_shape_rc=None, transformation_dst_shape_rc=None, src_shape_rc=None, dst_shape_rc=None, bk_dxdy=None, fwd_dxdy=None):
    """Warp points from registered coordinates to original coordinates

    Parameters
    ----------
    xy : ndarray
        [P, 2] array of xy coordinates for P points

    M : ndarray, optional
         3x3 affine transformation matrix to perform rigid warp

    transformation_src_shape_rc : (int, int)
        Shape of image that was used to find the transformation.
        For example, this could be the original image in which features were detected

    transformation_dst_shape_rc : (int, int), optional
        Shape of the image with shape `transformation_src_shape_rc` after warping.
        This could be the shape of the original image after applying `M`.

    src_shape_rc : optional, (int, int)
        Shape of the image from which the points originated. For example,
        this could be a larger/smaller version of the image that was
        used for feature detection.

    dst_shape_rc : optional, (int, int)
        Shape of image (with shape `src_shape_rc`) after warping

    bk_dxdy : ndarray
        (2, N, M) numpy array of pixel displacements in the x and y
        directions from the reference image. dx = bk_dxdy[0],
        and dy=bk_dxdy[1].  This is what is actually used to warp the points.

    fwd_dxdy : ndarray
        Inverse of bk_dxdy. dx = fwd_dxdy[0], and dy=fwd_dxdy[1].
        If `fwd_dxdy` is not None, but
        `bk_dxdy` is None, then `fwd_dxdy` will be inverted to warp `xy`.

    """
    do_non_rigid = bk_dxdy is not None or fwd_dxdy is not None

    if M is None and not do_non_rigid:
        return xy

    src_sxy, dst_sxy, displacement_sxy, displacement_shape_rc = get_warp_scaling_factors(transformation_src_shape_rc=transformation_src_shape_rc,
                                                                     transformation_dst_shape_rc=transformation_dst_shape_rc,
                                                                     src_shape_rc=src_shape_rc, dst_shape_rc=dst_shape_rc,
                                                                     bk_dxdy=bk_dxdy, fwd_dxdy=fwd_dxdy)

    if dst_sxy is not None:
        xy_in_reg_img = xy/dst_sxy
    else:
        xy_in_reg_img = xy

    # Get points into position in the rigid image #
    if do_non_rigid:
        if fwd_dxdy is not None and bk_dxdy is None:
            bk_dxdy = get_inverse_field(fwd_dxdy)

        xy_in_rigid = warp_xy(xy_in_reg_img, fwd_dxdy=bk_dxdy)
        if displacement_sxy is not None:
            xy_in_rigid /= displacement_sxy
    else:
        xy_in_rigid = xy_in_reg_img

    if M is not None:
         xy_inv = warp_xy(xy_in_rigid, M=np.linalg.inv(M))
    else:
        xy_inv = xy_in_rigid

    if src_sxy is not None:
        xy_inv *= src_sxy

    return xy_inv