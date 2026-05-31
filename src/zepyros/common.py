import numpy as np
import sys

from numba import jit
from scipy.special import gammaln


def rotate_matrix(cos, sin, axis):
    """
    Given sine and cosine of the angle and the axis of rotation returns
    the rotation matrix

    Parameters
    ----------
    `cos`: real
        the cosine of the angle
    `sin`: real
        the sine of the angle
    `axis`: integer
        the axis of rotation: 0 for x, 1 for y and 2 for z

    Returns
    -------
    matrix
        the rotation matrix
    """
    rotated_matrix = np.zeros((3, 3))
    if axis == 0:
        rotated_matrix[0, 0] = 1
        rotated_matrix[1, 1] = cos
        rotated_matrix[1, 2] = -sin
        rotated_matrix[2, 1] = sin
        rotated_matrix[2, 2] = cos
    if axis == 1:
        rotated_matrix[1, 1] = 1
        rotated_matrix[0, 0] = cos
        rotated_matrix[0, 2] = sin
        rotated_matrix[2, 0] = -sin
        rotated_matrix[2, 2] = cos
    if axis == 2:
        rotated_matrix[2, 2] = 1
        rotated_matrix[0, 0] = cos
        rotated_matrix[1, 0] = sin
        rotated_matrix[0, 1] = -sin
        rotated_matrix[1, 1] = cos
    return rotated_matrix


def rotate_patch(patch, v_start, v_z, pin):
    """
    Rotates a set of points by an angle defined by a given unit vector
    and the z axis with a positive or negative direction

    Parameters
    ----------
    `patch`: ndarray
        array of points that make up the portion of the surface to orient.
        If the array contains more than three columns, only the first three will be selected
    `v_start`: ndarray
        array of shape (1, 3) which indicates the initial unit vector to orient
        and make coincide with the z axis
    `v_z`: ndarray
        array of shape (1, 3) indicating the positive (0, 0, 1)
        or negative (0, 0, -1) oriented z-axis unit vector
    `pin`: ndarray
        array of shape (1, 3) indicating the origin of ``v_start``

    Return
    ------
    `tuple`
        # TODO: add return docs
    """
    xy = True
    yz = True
    # xz = True   # unused

    _ESP_ = 1e-10

    patch_trans = np.asarray(patch)[:, :3]
    v_start = np.asarray(v_start).reshape(-1)[:3]
    pin = np.asarray(pin).reshape(-1)[:3]
    v_z = np.asarray(v_z).reshape(-1)[0]

    patch_trans = patch_trans - pin

    # defining rotating vectors
    r_z = np.array([0, 0, v_z])
    r_vn = v_start.copy()  # np.mean(normal_v, axis=0) #nter_atom_pos[1, :]

    p = np.abs(1 - r_z.dot(r_vn) / np.sqrt((r_z.dot(r_z)) * (r_vn.dot(r_vn))))

    r_vn1 = r_vn.copy()

    while p > _ESP_:

        r1 = np.array([r_vn1[0], r_vn1[1]])
        # r2 = np.array([r_z[0], r_z[1]])   # unused
        r1 /= np.sqrt(r1.dot(r1))

        cos_theta = r1[1] / np.sqrt(r1.dot(r1))  # (r1[0]*r2[0] + r1[1]*r2[1])/r1.dot(r1)
        sin_theta = r1[0] / np.sqrt(r1.dot(r1))  # (r1[0]*r2[1] - r1[1]*r2[0])/r1.dot(r1)

        r = rotate_matrix(cos_theta, sin_theta, 2)
        if xy:
            patch_trans = np.dot(patch_trans, r.T)
            r_vn1 = np.dot(r, r_vn1)

        # y-z plane
        r1 = np.array([r_vn1[1], r_vn1[2]])
        # r2 = np.array([r_z[1], r_z[2]])   # unused
        r1 /= np.sqrt(r1.dot(r1))

        if v_z > 0:
            cos_theta = r1[1]
            sin_theta = r1[0]
        else:
            cos_theta = -r1[1]
            sin_theta = -r1[0]

        r = rotate_matrix(cos_theta, sin_theta, 0)
        if yz:
            patch_trans = np.dot(patch_trans, r.T)
            r_vn1 = np.dot(r, r_vn1)

        p = np.abs(1 - r_z.dot(r_vn1) / np.sqrt((r_z.dot(r_z)) * (r_vn1.dot(r_vn1))))
    return r_vn1, patch_trans


@jit(nopython=True, cache=True)
def _isolate_surfaces_jit(surface, min_d2):
    n_points = surface.shape[0]
    labels = np.zeros(n_points, dtype=np.int8)
    stack = np.empty(n_points, dtype=np.int64)
    lab = 2

    for seed in range(n_points):
        if labels[seed] != 0:
            continue

        stack_top = 0
        stack[stack_top] = seed
        stack_top += 1
        labels[seed] = lab

        while stack_top > 0:
            stack_top -= 1
            i = stack[stack_top]

            for j in range(n_points):
                if labels[j] == 0:
                    d2 = (surface[j, 0] - surface[i, 0]) ** 2 + (surface[j, 1] - surface[i, 1]) ** 2 + (surface[j, 2] - surface[i, 2]) ** 2
                    if d2 < min_d2:
                        labels[j] = lab
                        stack[stack_top] = j
                        stack_top += 1

        lab += 1

    return labels


@jit(nopython=True, cache=True)
def _contact_points_jit(list_1, list_2, thresh2):
    l1 = list_1.shape[0]
    l2 = list_2.shape[0]

    # flags and mins per element to avoid duplicates and extra sorting
    contact_1_flag = np.zeros(l1, dtype=np.bool_)
    contact_1_min = np.empty(l1, dtype=np.float64)
    for i in range(l1):
        contact_1_min[i] = np.inf

    contact_2_flag = np.zeros(l2, dtype=np.bool_)
    contact_2_min = np.empty(l2, dtype=np.float64)
    for j in range(l2):
        contact_2_min[j] = np.inf

    for i in range(l1):
        for j in range(l2):
            d2 = (list_1[i, 0] - list_2[j, 0]) ** 2 + (list_1[i, 1] - list_2[j, 1]) ** 2 + (list_1[i, 2] - list_2[j, 2]) ** 2
            if d2 < thresh2:
                contact_1_flag[i] = True
                if d2 < contact_1_min[i]:
                    contact_1_min[i] = d2
                contact_2_flag[j] = True
                if d2 < contact_2_min[j]:
                    contact_2_min[j] = d2

    contact_1_count = 0
    for i in range(l1):
        if contact_1_flag[i]:
            contact_1_count += 1

    contact_2_count = 0
    for j in range(l2):
        if contact_2_flag[j]:
            contact_2_count += 1

    contact_1 = np.empty((contact_1_count, 4), dtype=np.float64)
    contact_2 = np.empty((contact_2_count, 4), dtype=np.float64)

    c1pos = 0
    for i in range(l1):
        if contact_1_flag[i]:
            contact_1[c1pos, 0] = list_1[i, 0]
            contact_1[c1pos, 1] = list_1[i, 1]
            contact_1[c1pos, 2] = list_1[i, 2]
            contact_1[c1pos, 3] = contact_1_min[i]
            c1pos += 1

    c2pos = 0
    for j in range(l2):
        if contact_2_flag[j]:
            contact_2[c2pos, 0] = list_2[j, 0]
            contact_2[c2pos, 1] = list_2[j, 1]
            contact_2[c2pos, 2] = list_2[j, 2]
            contact_2[c2pos, 3] = contact_2_min[j]
            c2pos += 1

    return contact_1, contact_2


@jit(nopython=True, cache=True)
def _contact_points_with_index_jit(list_1, list_2, thresh2):
    l1 = list_1.shape[0]
    l2 = list_2.shape[0]

    contact_1_flag = np.zeros(l1, dtype=np.bool_)
    contact_1_min = np.empty(l1, dtype=np.float64)
    for i in range(l1):
        contact_1_min[i] = np.inf

    contact_2_flag = np.zeros(l2, dtype=np.bool_)
    contact_2_min = np.empty(l2, dtype=np.float64)
    for j in range(l2):
        contact_2_min[j] = np.inf

    for i in range(l1):
        for j in range(l2):
            d2 = (list_1[i, 0] - list_2[j, 0]) ** 2 + (list_1[i, 1] - list_2[j, 1]) ** 2 + (list_1[i, 2] - list_2[j, 2]) ** 2
            if d2 < thresh2:
                contact_1_flag[i] = True
                if d2 < contact_1_min[i]:
                    contact_1_min[i] = d2
                contact_2_flag[j] = True
                if d2 < contact_2_min[j]:
                    contact_2_min[j] = d2

    contact_1_count = 0
    for i in range(l1):
        if contact_1_flag[i]:
            contact_1_count += 1

    contact_2_count = 0
    for j in range(l2):
        if contact_2_flag[j]:
            contact_2_count += 1

    contact_1 = np.empty((contact_1_count, 4), dtype=np.float64)
    contact_2 = np.empty((contact_2_count, 4), dtype=np.float64)
    list_index_1 = np.empty(contact_1_count, dtype=np.int64)
    list_index_2 = np.empty(contact_2_count, dtype=np.int64)

    c1pos = 0
    for i in range(l1):
        if contact_1_flag[i]:
            contact_1[c1pos, 0] = list_1[i, 0]
            contact_1[c1pos, 1] = list_1[i, 1]
            contact_1[c1pos, 2] = list_1[i, 2]
            contact_1[c1pos, 3] = contact_1_min[i]
            list_index_1[c1pos] = i
            c1pos += 1

    c2pos = 0
    for j in range(l2):
        if contact_2_flag[j]:
            contact_2[c2pos, 0] = list_2[j, 0]
            contact_2[c2pos, 1] = list_2[j, 1]
            contact_2[c2pos, 2] = list_2[j, 2]
            contact_2[c2pos, 3] = contact_2_min[j]
            list_index_2[c2pos] = j
            c2pos += 1

    return contact_1, contact_2, list_index_1, list_index_2


def flip_matrix(mat, axis):
    """
    Flips a matrix based on its axis

    Parameters
    ----------
    `mat`: ndarray
        the matrix to flip
    `axis`: int
        the axis about which to flip the ndarray (0 for x, 1 for y)

    Return
    ------
    matrix
        the flipped matrix with respect the axis

    Examples
    --------
    >>> mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> mat
    array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
    >>> flip_matrix(mat, 0)
    array([[7, 8, 9],
       [4, 5, 6],
       [1, 2, 3]])
    >>> flip_matrix(mat, 1)
    array([[3, 2, 1],
       [6, 5, 4],
       [9, 8, 7]])
    """
    if not hasattr(mat, 'ndim'):
        m = np.asarray(mat)
    indexer = [slice(None)] * mat.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, mat.ndim))
    return mat[tuple(indexer)]


def isolate_surfaces(surface, min_d=1.):
    """
    Given a surface of x, y, z points, form groups of points that are
    at a distance less than a given threshold

    Parameters
    ----------
    `surface`: ndarray
        array of points where each row is a point in space.
        If the number of columns is greater than three,
        only the first three columns will be used
    `min_d`: float
        the minimum distance at which two points must be located
        to be grouped in the same cluster.
        The unit of measurement depends on that of the points in `surface`

    Return
    ------
    array
        list of integer (starts from 2), where each integer represents
        the label of a specific group. The i-th element indicates the group
        to which the i-th point of the initial surface belongs

    Examples
    --------
    >>> mat = np.array([[4.5, 4.5, 4.5], [1.5, 1.5, 1.5], [2., 1., 1.5], [5., 5., 5.], [1., 1., 1.]])
    >>> mat
    array([[4.5, 4.5, 4.5],
           [1.5, 1.5, 1.5],
           [2. , 1. , 1.5],
           [5. , 5. , 5. ],
           [1. , 1. , 1. ]])
    >>> isolate_surfaces(mat, 1)
    array([2, 3, 3, 2, 3], dtype=int8)
    """
    surface = np.asarray(surface)
    if surface.shape[1] > 3:
        surface = surface[:, :3]

    return _isolate_surfaces_jit(surface, min_d ** 2)


def _find_border(new_plane_ab):
    """
    This function finds the border of a figure in the plane...
    """
    # TODO: add documentation. DEPRECATED
    a, b = np.shape(new_plane_ab)
    p_h = np.ones((a, b))
    p_v = np.ones((a, b))

    index = np.arange(0, a)

    for i in range(a):

        # horizontal
        tmp = new_plane_ab[i, :]
        lr = tmp != 0
        if len(index[lr]) > 0:
            pos_1 = index[lr][0]
            pos_2 = index[lr][-1]

            p_h[i, :pos_1] = 0
            p_h[i, (pos_2 + 1):] = 0
        else:
            p_h[i, :] = 0

        # vertical
        tmp = new_plane_ab[:, i]
        lr = tmp != 0
        if len(index[lr]) > 0:
            pos_1 = index[lr][0]
            pos_2 = index[lr][-1]

            p_v[:pos_1, i] = 0
            p_v[(pos_2 + 1):, i] = 0
        else:
            p_v[:, i] = 0

    return p_v * p_h


def contact_points(list_1, list_2, thresh):
    """
    Given two lists of points, find the points in each list that are closer
    than a given threshold to at least one point in the other list

    Parameters
    ----------
    `list_1`: ndarray
        array of points in space. If the array contains more than three columns,
        only the first three will be selected
    `list_2`: ndarray
        array of points in space. If the array contains more than three columns,
        only the first three will be selected
    `thresh`: float
        the minimum distance at which two points in different lists must be located
        to be considered in contact.
        The unit of measurement depends on that of the points in ``list_1`` and ``list_2``

    Return
    ------
    `tuple`
        - the points in ``list_1`` in contact with ``list_2`` (`ndarray`)
        - the points in ``list_2`` in contact with ``list_1`` (`ndarray`)
    """
    list_1 = np.asarray(list_1)
    list_2 = np.asarray(list_2)
    if list_1.shape[1] > 3:
        list_1 = list_1[:, :3]
    if list_2.shape[1] > 3:
        list_2 = list_2[:, :3]

    contact_1, contact_2 = _contact_points_jit(list_1, list_2, thresh ** 2)
    # `_contact_points_jit` already returns unique `contact_2` rows with
    # the minimum squared distance in column 3. Just ensure empty shape when no hits.
    if contact_2.shape[0] == 0:
        contact_2 = np.empty((0, 4), dtype=np.float64)
    else:
        # sort lexicographically by x,y,z to reproduce np.unique(..., axis=0) ordering
        order = np.lexsort((contact_2[:, 2], contact_2[:, 1], contact_2[:, 0]))
        contact_2 = contact_2[order]

    return contact_1, contact_2


def _contact_points(list_1, list_2, thresh):
    """
    Given two lists of points, find the points in each list that are closer
    than a given threshold to at least one point in the other list

    Parameters
    ----------
    `list_1`: ndarray
        array of points in space. If the array contains more than three columns,
        only the first three will be selected
    `list_2`: ndarray
        array of points in space. If the array contains more than three columns,
        only the first three will be selected
    `thresh`: float
        the minimum distance at which two points in different lists must be located
        to be considered in contact.
        The unit of measurement depends on that of the points in ``list_1`` and ``list_2``

    Return
    ------
    `tuple`
        - the points in ``list_1`` in contact with ``list_2`` (`ndarray`)
        - the points in ``list_2`` in contact with ``list_1`` (`ndarray`)
        - the indexes of the points in ``list_1`` in contact with ``list_2`` (`ndarray`)
        - the indexes of the points in ``list_2`` in contact with ``list_1`` (`ndarray`)
    """
    list_1 = np.asarray(list_1)
    list_2 = np.asarray(list_2)
    if list_1.shape[1] > 3:
        list_1 = list_1[:, :3]
    if list_2.shape[1] > 3:
        list_2 = list_2[:, :3]

    contact_1, contact_2, list_index_1, list_index_2 = _contact_points_with_index_jit(list_1, list_2, thresh ** 2)
    # `_contact_points_with_index_jit` returns unique index arrays and
    # `contact_2` already contains min squared distances in column 3.
    if contact_2.shape[0] == 0:
        contact_2 = np.empty((0, 4), dtype=np.float64)
    else:
        # match previous behavior: unique(contact_2, axis=0) sorts lexicographically
        order = np.lexsort((contact_2[:, 2], contact_2[:, 1], contact_2[:, 0]))
        contact_2 = contact_2[order]

    # previous code did `list_index_2 = np.unique(list_index_2)` which returns sorted indices
    if list_index_2.shape[0] > 0:
        list_index_2 = np.sort(list_index_2)

    return contact_1, contact_2, list_index_1, list_index_2


def _build_cone(z_max, n_disk):
    # TODO: maybe deprecated? Only plot
    dz = z_max / float(n_disk)
    z = 0

    n = 100
    res = [0, 0, 0]
    rad = np.linspace(0, 2 * np.pi, n)
    for i in range(n_disk):
        z += dz
        x = z * np.cos(rad)
        y = z * np.sin(rad)

        res = np.row_stack([res, np.column_stack([x, y, np.ones(n) * z])])
    return res


def _concatenate_fig_plots(list_):
    # TODO: add documentation. Maybe deprecated?
    l = len(list_)
    res = list_[0]

    n, tmp = np.shape(list_[0])

    col_list = np.linspace(-100, 100, l)
    col = np.ones(n) * col_list[0]

    if l > 1:
        for i in range(1, l):
            res = np.row_stack([res, list_[i]])
            n, tmp = np.shape(list_[i])
            col = np.concatenate([col, np.ones(n) * col_list[i]])
    return res, col


def _fix_bridge_real_bs(patch_template, patch_target, d_pp):
    """
    This function isolates the different groups of points in two given sets (patches)
    according to a cutoff distance Dpp.
    It associates each group to the closest group of the other set and returns
    a list of matched patches.
    """
    # TODO: add documentation.
    # processing  patches to remove islands
    # template
    index_template_bd_ = isolate_surfaces(patch_template, d_pp)
    val_template, counts_template = np.unique(index_template_bd_, return_counts=True)

    # target
    index_target_bd_ = isolate_surfaces(patch_target, d_pp)
    val_target, counts_target = np.unique(index_target_bd_, return_counts=True)

    # creating matrix of number of points in each group and label returned by isolate_surface func
    # columns are ordered from the biggest to the smallest patch.
    tmp = np.row_stack([counts_template, val_template])
    s_c_template = flip_matrix(tmp[:, np.argsort(tmp[0, :])], axis=1)

    tmp = np.row_stack([counts_target, val_target])
    s_c_target = flip_matrix(tmp[:, np.argsort(tmp[0, :])], axis=1)

    # finding center of mass of each group
    cm_template = []
    cm_target = []
    for el1 in s_c_template[1, :]:
        tmp = patch_template[index_template_bd_ == el1]
        cm_template.append(np.mean(tmp[:, :3], axis=0))
    for el2 in s_c_target[1, :]:
        tmp = patch_target[index_target_bd_ == el2]
        cm_target.append(np.mean(tmp[:, :3], axis=0))

    l_template = np.shape(s_c_template)[1]
    l_target = np.shape(s_c_target)[1]

    # computing distance matrix between the centers of the groups intra sets
    d = np.zeros((l_template, l_target))
    for i in range(l_template):
        for j in range(l_target):
            d[i, j] = np.sum((np.array(cm_template[i]) - np.array(cm_target[j])) ** 2)

    # associating groups according to the minimal distance
    index_template = []
    index_target = []
    l_min = np.min([l_target, l_template])
    for i in range(l_min):
        if l_min == l_template:
            x = np.where(d[i, :] == np.min(d[i, :]))[0][0]
            index_template.append(i)
            index_target.append(x)
        else:
            x = np.where(d[:, i] == np.min(d[:, i]))[0][0]
            index_template.append(x)
            index_target.append(i)

    patch_template_list = []
    patch_target_list = []

    lab_template = s_c_template[1, index_template]
    lab_target = s_c_target[1, index_target]

    # defining list of matched patches
    for i in range(l_min):
        patch_template_list.append(patch_template[index_template_bd_ == lab_template[i]])
        patch_target_list.append(patch_target[index_target_bd_ == lab_target[i]])

    return patch_template_list, patch_target_list


def _isolate_isosurface(my_prot, min_v, max_v):
    # TODO: maybe deprecated?
    # This function groups points nearer than minD

    _DEB_ = 0

    lx, ly, lz = np.shape(my_prot)

    prot = np.copy(my_prot)
    mask = np.logical_and(prot >= min_v, prot <= max_v)
    prot[:, :, :] = 0
    prot[mask] = 1.
    prot_label = np.copy(prot)

    # starting from label = 2
    lab = 2

    # defining probe points
    tmp = np.zeros((3, 3, 3))
    x, y, z = np.where(tmp == 0)
    x = x - 1
    y = y - 1
    z = z - 1

    # computing number of points without label
    n_left = np.sum(prot != 0)

    # starting iterating over different surfaces
    while n_left > 0:
        count = 1
        pos__ = np.where(prot != 0)
        if _DEB_:
            print("pos__", np.shape(pos__))

        # seeding: first unlabeled point takes lab label.
        prot_label[pos__[0][0], pos__[1][0], pos__[2][0]] = lab
        prot[pos__[0][0], pos__[1][0], pos__[2][0]] = lab

        if _DEB_:
            print("prot", prot)

        # iterating to find points belonging to the same surface...
        while count > 0:
            count = 0
            pos_s = np.where(prot == lab)
            if len(pos_s) == 0:
                break
            # creating mask for points still to be processed
            mask = np.logical_and(prot > 0, prot != lab)
            if _DEB_:
                print("l", np.shape(pos_s))
            for i in range(np.shape(pos_s)[1]):
                if _DEB_:
                    print("pos", i, np.shape(pos_s))
                if np.shape(pos_s)[1] == 1:
                    xxxx = pos_s[0]
                    yyyy = pos_s[1]
                    zzzz = pos_s[2]
                else:
                    xxxx = pos_s[0][i]
                    yyyy = pos_s[1][i]
                    zzzz = pos_s[2][i]
                x_ = x + xxxx
                y_ = y + yyyy
                z_ = z + zzzz

                mask_x = np.logical_and(x_ >= 0, x_ < lx)
                mask_y = np.logical_and(y_ >= 0, y_ < ly)
                mask_z = np.logical_and(z_ >= 0, z_ < lz)

                mask_xyz = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

                x_ = x_[mask_xyz]
                y_ = y_[mask_xyz]
                z_ = z_[mask_xyz]
                if _DEB_:
                    print("xyz", x_, y_, z_)

                for j in range(len(x_)):
                    if prot[x_[j], y_[j], z_[j]] == 1:
                        prot[x_[j], y_[j], z_[j]] = lab
                        prot_label[x_[j], y_[j], z_[j]] = lab
                        count += 1

                if _DEB_:
                    print("prot_2", prot)

                # removing processed point from system
                prot[xxxx, yyyy, zzzz] = 0
                mm = np.logical_and(prot > 0, prot != lab)

                # creating mask for  point still to be processed
                mmm = np.logical_and(mm, mask)

                mask = np.logical_and(prot > 0, prot != lab)
                if _DEB_:
                    print("prot_c", prot)

        # creating a new label
        lab += 1
        # looking for how many points still to be processed
        n_left = np.sum(prot != 0)
        sys.stderr.write("\rleft %d" % n_left)
        sys.stderr.flush()
    return prot_label


def log10_factorial(n):
    """
    Compute ``log(n!)`` with the base 10 logarithm

    Parameters
    ----------
    `n`: int
        Input values. If n < 0, the return value is 0

    Return
    ------
    real
        Factorial of log10(n): ``log(n!) = log(n) + log(n-1) + ... + log(1)``

    Examples
    --------
    >>> log10_factorial(10)
    6.559763032876794
    """
    n = np.asarray(n)
    output = np.zeros_like(n, dtype=np.float64)
    mask = n > 1
    if np.any(mask):
        output[mask] = gammaln(n[mask] + 1) / np.log(10)
    if output.ndim == 0:
        return output.item()
    return output
