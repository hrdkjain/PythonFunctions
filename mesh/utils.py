from scipy.io import loadmat
import numpy as np
import os
import torch
from tqdm import tqdm

from torch.distributions.categorical import Categorical

def maticomesh2numpy(mat_file):

    matdict = loadmat(mat_file)
    planar_image = matdict["planar_image"]
    planar_image = np.moveaxis(planar_image.astype(dtype=np.float32), [0, 1], [1, 0])
    inv_idx_orig = None
    if 'inv_idx_orig' in matdict.keys():
        inv_idx_orig = matdict["inv_idx_orig"]
        inv_idx_orig = np.moveaxis(inv_idx_orig.astype(dtype=np.int) - 1, [0, 1], [1, 0])
    inv_idx_sampling = None
    if 'inv_idx_sampling' in matdict.keys():
        inv_idx_sampling = matdict["inv_idx_sampling"]
        inv_idx_sampling = np.moveaxis(inv_idx_sampling.astype(dtype=np.int) - 1, [0, 1], [1, 0])
    inv_offset = None
    if 'inv_offset' in matdict.keys():
        inv_offset = matdict["inv_offset"]
        inv_offset = np.moveaxis(inv_offset.astype(dtype=np.float32), [0, 1], [1, 0])

    return planar_image, inv_idx_orig, inv_idx_sampling, inv_offset


def normalize_pos_unitsphere(points_inout):
    centroid = np.mean(points_inout.reshape(3, -1), axis=1).reshape([3] + [1]*(points_inout.ndim - 1))
    points_inout -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(np.abs(points_inout) ** 2, axis=0)))
    points_inout /= furthest_distance


def get_normalization_unitsphere(points_in):
    centroid = np.mean(points_in, axis=1, keepdims=True)
    furthest_distance = np.max(np.sqrt(np.sum(np.abs(points_in - centroid) ** 2, axis=0)))
    return centroid, furthest_distance


def mesh_reconstruct_orig(samples, mapping):
    pass


def convert_normalize_meshes(in_dir, out_dir):
    if not os.path.exists(in_dir) or not os.path.exists(out_dir):
        print("in_dir or out_dir do not exist")
        return

    files = os.listdir(in_dir)
    for i in tqdm(range(len(files))):
        f = files[i]
        if f.endswith('.mat'):
            # load file
            f_in_absolute = os.path.join(in_dir, f)
            data, inv_idx_orig, inv_idx_sampling, inv_offset = maticomesh2numpy(f_in_absolute)

            if inv_idx_orig is not None:
                # construct unique vertices
                inv_idx = torch.cat((torch.from_numpy(inv_idx_orig), torch.from_numpy(inv_idx_sampling)), dim=0)
                cnt = torch.from_numpy(np.bincount(inv_idx_orig[0]))
                v_orig = torch.from_numpy(data[:3, inv_idx_sampling].squeeze() - inv_offset)
                x = torch.sparse.FloatTensor(inv_idx, v_orig[0])
                y = torch.sparse.FloatTensor(inv_idx, v_orig[1])
                z = torch.sparse.FloatTensor(inv_idx, v_orig[2])
                w = torch.stack((x, y, z), dim=0)
                v_unique = torch.sparse.sum(w, dim=2).to_dense() / cnt

                # normalize
                centroid, furthest_distance = get_normalization_unitsphere(v_unique.numpy())
                data[:3] -= centroid
                data[:3] /= furthest_distance
                inv_offset /= furthest_distance
            else:
                # normalize data inplace
                normalize_pos_unitsphere(data[:3])

            if np.any(np.isnan(data)):
                print("Error: NaNs in file {0}".format(f))
                continue

            # store npz file
            f_out_absolute = os.path.join(out_dir, f[:-3] + 'npz')
            np.savez(f_out_absolute,
                     data=data, inv_idx_orig=inv_idx_orig, inv_idx_sampling=inv_idx_sampling, inv_offset=inv_offset)


def mesh_p2p_dist(mesh_0, mesh_1):
    with torch.no_grad():
        return torch.sqrt(torch.sum((mesh_0 - mesh_1) ** 2, dim=1))


def mesh_avg_p2p_dist(mesh_0, mesh_1):
    with torch.no_grad():
        p2p_dist = mesh_p2p_dist(mesh_0, mesh_1)
        return torch.mean(p2p_dist.view(p2p_dist.shape[0], -1), dim=1)


def compute_face_normals(v, f, eps=1e-10):
    # get vertices for face positions 0, 1, 2
    a = torch.index_select(v, dim=1, index=f[:, 0])
    b = torch.index_select(v, dim=1, index=f[:, 1])
    c = torch.index_select(v, dim=1, index=f[:, 2])

    # compute unnormalized (area weighted) face normals
    f_normals = torch.cross(b - a, c - a, dim=2)

    # normalize
    magnitude = torch.clamp(torch.norm(f_normals, 2, dim=2, keepdim=True), eps)
    f_normals = torch.div(f_normals, magnitude)

    return f_normals


def compute_vertex_normals(v, f, weight_face_area=True, eps=1e-10):
    # get vertices for face positions 0, 1, 2
    a = torch.index_select(v, dim=1, index=f[:, 0])
    b = torch.index_select(v, dim=1, index=f[:, 1])
    c = torch.index_select(v, dim=1, index=f[:, 2])

    # compute unnormalized (area weighted) face normals
    f_normals = torch.cross(b - a, c - a, dim=2)
    if not weight_face_area:
        # normalize
        #magnitude = torch.clamp(torch.sqrt(torch.sum(torch.pow(f_normals, 2), dim=2, keepdim=True)), eps)
        magnitude = torch.clamp(torch.norm(f_normals, 2, dim=2, keepdim=True), eps)
        f_normals = torch.div(f_normals, magnitude)

    # accumulate vertex normals from face normals
    v_normals = torch.zeros_like(v)
    v_normals = v_normals.index_add(1, f[:, 0], f_normals)
    v_normals = v_normals.index_add(1, f[:, 1], f_normals)
    v_normals = v_normals.index_add(1, f[:, 2], f_normals)

    # normalize
    # magnitude = torch.clamp(torch.sqrt(torch.sum(torch.pow(v_normals, 2), dim=2, keepdim=True)), eps)
    magnitude = torch.clamp(torch.norm(v_normals, 2, dim=2, keepdim=True), eps)
    v_normals = torch.div(v_normals, magnitude)

    return v_normals


def compute_laplacian(v, adj_sparse):
    # adapted from
    # https://kaolin.readthedocs.io/en/latest/_modules/kaolin/rep/Mesh.html#Mesh.compute_laplacian
    neighbor_sum = torch.sparse.mm(adj_sparse, v) - v
    neighbor_num = torch.sparse.sum(adj_sparse, dim=1).to_dense().view(-1, 1) - 1
    neighbor_num[neighbor_num == 0] = 1
    neighbor_num = (1. / neighbor_num).view(-1, 1)

    neighbor_sum = neighbor_sum * neighbor_num
    lap = v - neighbor_sum
    return lap

def compute_laplacian_batch(v, adj_sparse):
    # adapted from
    # https://kaolin.readthedocs.io/en/latest/_modules/kaolin/rep/Mesh.html#Mesh.compute_laplacian
    neighbor_sum = torch.stack([torch.sparse.mm(adj_sparse, v[i]) - v[i] for i in range(v.shape[0])])
    # neighbor_sum = torch.sparse.mm(adj_sparse, v) - v
    neighbor_num = torch.sparse.sum(adj_sparse, dim=1).to_dense() - 1
    neighbor_num[neighbor_num == 0] = 1
    neighbor_num = (1. / neighbor_num)

    neighbor_sum = neighbor_sum * neighbor_num.view(1, -1, 1)
    lap = v - neighbor_sum
    return lap


def cotangent_vectors(v0, v1, eps=1e-10):

    # nominator is: dot(v0, v1)^2
    nom = torch.sum(v0*v1, dim=2, keepdim=True)

    # denominator is || v0 x v1 ||^2
    denom = torch.cross(v0, v1, dim=2)
    denom = torch.norm(denom, p=2, dim=2, keepdim=True)
    if eps:
        denom = torch.clamp(denom, eps)

    # compute cotangent as nom/denom
    cotan = torch.div(nom, denom)
    return cotan


def compute_laplaceBeltrami(v, f, eps=1e-10):
    # get vertices
    v0 = torch.index_select(v, dim=1, index=f[:, 0])
    v1 = torch.index_select(v, dim=1, index=f[:, 1])
    v2 = torch.index_select(v, dim=1, index=f[:, 2])

    # get triangle edges as vectors
    v01 = v1 - v0
    v02 = v2 - v0
    v12 = v2 - v1

    # get cotangens for each corner
    cot0 = cotangent_vectors(v01, v02)
    cot1 = cotangent_vectors(v12, -v01)
    cot2 = cotangent_vectors(-v12, -v02)

    # compute laplace beltrami elements
    lb01 = cot2 * v01
    lb02 = cot1 * v02
    lb12 = cot0 * v12

    lap = torch.zeros_like(v)
    lap = lap.index_add(dim=1, index=f[:, 0], source=-lb01)
    lap = lap.index_add(dim=1, index=f[:, 0], source=-lb02)
    lap = lap.index_add(dim=1, index=f[:, 1], source=-lb12)
    lap = lap.index_add(dim=1, index=f[:, 1], source=lb01)
    lap = lap.index_add(dim=1, index=f[:, 2], source=lb02)
    lap = lap.index_add(dim=1, index=f[:, 2], source=lb12)

    # compute face areas
    f_area = torch.cross(v01, v02, dim=2)
    f_area = torch.norm(f_area, p=2, dim=2, keepdim=True) / 2.0
    f_area = f_area / 3

    # compute vertex areas
    v_area = v.new_zeros([v.shape[0], v.shape[1], 1])
    v_area = v_area.index_add(dim=1, index=f[:, 0], source=f_area)
    v_area = v_area.index_add(dim=1, index=f[:, 1], source=f_area)
    v_area = v_area.index_add(dim=1, index=f[:, 2], source=f_area)

    lap = lap / (v_area * 2.0)

    return lap



    # f_normals = torch.cross(b - a, c - a, dim=2)




def compute_adjacency_matrix_sparse(vert_len, f):
    # adapted from
    # https://kaolin.readthedocs.io/en/latest/_modules/kaolin/rep/TriangleMesh.html#TriangleMesh.compute_adjacency_matrix_sparse
    v1 = f[:, 0].view(-1, 1)
    v2 = f[:, 1].view(-1, 1)
    v3 = f[:, 2].view(-1, 1)

    identity_indices = torch.arange(vert_len, dtype=torch.long).view(-1, 1).to(v1.device)
    identity = torch.cat((identity_indices, identity_indices), dim=1).to(v1.device)
    identity = torch.cat((identity, identity))

    i_1 = torch.cat((v1, v2), dim=1)
    i_2 = torch.cat((v1, v3), dim=1)

    i_3 = torch.cat((v2, v1), dim=1)
    i_4 = torch.cat((v2, v3), dim=1)

    i_5 = torch.cat((v3, v2), dim=1)
    i_6 = torch.cat((v3, v1), dim=1)
    indices = torch.cat(
        (identity, i_1, i_2, i_3, i_4, i_5, i_6), dim=0).t()
    values = torch.ones(indices.shape[1]).to(indices.device) * .5
    adj_sparse = torch.sparse.FloatTensor(indices, values, torch.Size([vert_len, vert_len]))
    return adj_sparse


def sample_faces_batch(vertices: torch.Tensor, faces: torch.Tensor, num_samples: int, eps: float = 1e-10):
    # batch = vertices.shape[0]
    # dist_uni = torch.distributions.Uniform(vertices.new_tensor([0.0] * batch), vertices.new_tensor([1.0] * batch))

    # get vertices for face positions 0, 1, 2
    a = torch.index_select(vertices, dim=1, index=faces[:, 0])
    b = torch.index_select(vertices, dim=1, index=faces[:, 1])
    c = torch.index_select(vertices, dim=1, index=faces[:, 2])

    # compute unnormalized (area weighted) face normals
    f_normals = torch.cross(b - a, c - a, dim=2)
    Areas = torch.sqrt(torch.sum(f_normals ** 2, 2)) # / 2.0 division by two not needed as areas will be normalized
    Areas = Areas / (torch.sum(Areas, 1, keepdim=True) + eps)

    # define descrete distribution w.r.t. face area ratios caluclated
    #cat_dist = torch.distributions.Categorical(Areas)
    cat_dist = Categorical(Areas)
    face_choices = cat_dist.sample([num_samples])
    # face_choices = cat_dist.sample([num_samples]).t()
    #
    # # from each face sample a point
    # select_faces = faces[face_choices] + \
    #                (torch.arange(vertices.shape[0], device=vertices.device) * vertices.shape[1]).view(-1, 1, 1)
    # select_faces = select_faces.contiguous().view(-1, 3)
    # vv = vertices.view(-1, 3)
    #
    # v0 = torch.index_select(vv, 0, select_faces[:, 0]).view(batch, num_samples, 3)
    # v1 = torch.index_select(vv, 0, select_faces[:, 1]).view(batch, num_samples, 3)
    # v2 = torch.index_select(vv, 0, select_faces[:, 2]).view(batch, num_samples, 3)
    # u = torch.sqrt(dist_uni.sample([num_samples])).t().unsqueeze(-1)
    # v = dist_uni.sample([num_samples]).t().unsqueeze(-1)
    # points = (1 - u) * v0 + (u * (1 - v)) * v1 + u * v * v2

    # return points, face_choices
    return face_choices

def sample_point_from_face_batch(vertices: torch.Tensor, faces: torch.Tensor, face_choices: torch.Tensor, eps: float = 1e-10):
    num_samples = face_choices.shape[1]
    batch = vertices.shape[0]
    features = vertices.shape[2]
    dist_uni = torch.distributions.Uniform(vertices.new_tensor([0] * batch), vertices.new_tensor([1] * batch))

    # from each face sample a point
    select_faces = faces[face_choices]
    v0 = torch.gather(vertices, 1, select_faces[:, :, 0].unsqueeze(-1).expand(-1, -1, features))
    v1 = torch.gather(vertices, 1, select_faces[:, :, 1].unsqueeze(-1).expand(-1, -1, features))
    v2 = torch.gather(vertices, 1, select_faces[:, :, 2].unsqueeze(-1).expand(-1, -1, features))
    u = torch.sqrt(dist_uni.sample([num_samples])).t().unsqueeze(-1)
    v = dist_uni.sample([num_samples]).t().unsqueeze(-1)
    points = (1 - u) * v0 + (u * (1 - v)) * v1 + u * v * v2
    return points



import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'


def disp_icomesh(v, f, v_colors, name, v_normals, sizeref):
    #v_c_normalized = v_colors / 255.
    fig = go.Figure(data=[go.Mesh3d(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=f[:, 0],
        j=f[:, 1],
        k=f[:, 2],
        flatshading=True,
       #vertexcolor=v_colors
    )
        , go.Cone(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        u=v_normals[:, 0],
        v=v_normals[:, 1],
        w=v_normals[:, 2],
        colorscale='Blues',
        sizemode="absolute",
        sizeref=sizeref)

    ])

    fig.update_layout(
        title=name,
        autosize=True,
        margin=go.layout.Margin(
            l=5,
            r=5,
            b=5,
            t=5,
            pad=4
        )
    )
    fig.show()






if __name__ == '__main__':
    # import plotly.graph_objects as go
    # import plotly.io as pio
    # pio.renderers.default = 'browser'
    #
    # testfile_in = "/home/jbo/masterthesis/code/IcosahedralCNN/examples/mesh/airplane_1_dlp.mat"
    # testfile_out = "/home/jbo/masterthesis/code/IcosahedralCNN/examples/mesh/airplane_1_dlp.npz"
    # # maticomesh2numpy(testfile_in, testfile_out)
    #
    # with np.load(testfile_out) as file:
    #     data = file['data']
    #
    # #fig = go.Figure(data=[go.Scatter3d(x=data[0].flatten(), y=data[1].flatten(), z=data[2].flatten(), mode='markers')])
    # #fig.show()
    #
    # normalize_pos_unitsphere(data[:3])
    #
    # fig = go.Figure(data=[go.Scatter3d(x=data[0].flatten(), y=data[1].flatten(), z=data[2].flatten(), mode='markers')])
    # fig.show()

    dir_airplanes_in = '/mnt/miranda/jbo/Dataset/airplane4/V128A_noise_gw_diag_none_001_Sph_I5'
    dir_airplanes_out = '/shared/data/airplane4/V128A_noise_gw_diag_none_001_Sph_I5'
    convert_normalize_meshes(dir_airplanes_in, dir_airplanes_out)

    # from icocnn.utils.ico_geometry import get_icosahedral_grid
    #
    # subdivisions = 5
    #
    # v, f = get_icosahedral_grid(subdivisions)
    # v = torch.from_numpy(v)
    # f = torch.from_numpy(f)
    #
    # v = v.permute(1, 0)
    # v = torch.cat((v.unsqueeze(0), v.unsqueeze(0)), dim=0)
    #
    #
    # v_normals = compute_vertex_normals(v, f, False)
    #
    # disp_icomesh(v[0].t(), f, None, "test", v_normals[0].t(), 1.0)
    # disp_icomesh(v[1].t(), f, None, "test", v_normals[1].t(), 1.0)

    # a = torch.index_select(v, dim=0, index=f[:, 0].flatten())
    # b = torch.index_select(v, dim=0, index=f[:, 1].flatten())
    # c = torch.index_select(v, dim=0, index=f[:, 2].flatten())
    #
    # # compute unnormalized face normals
    # f_normals = torch.cross(b - a, c - a, dim=1)
    #
    # sref = 0.001
    #
    # disp_icomesh(v, f, None, "test", f_normals, sref)
    #
    # # accumulate vertex normals from face normals
    # v_normals = torch.zeros_like(v)
    # v_normals = v_normals.index_add(0, f[:, 0], f_normals)
    # v_normals = v_normals.index_add(0, f[:, 1], f_normals)
    # v_normals = v_normals.index_add(0, f[:, 2], f_normals)
    #
    # sref = 0.1
    #
    # disp_icomesh(v, f, None, "test", v_normals, sref)
    #
    # magnitude = torch.sqrt(torch.sum(torch.pow(v_normals, 2), dim=1))
    # valid_inds = magnitude > 0
    # v_normals[valid_inds] = torch.div(v_normals[valid_inds], magnitude[valid_inds].unsqueeze(1))













