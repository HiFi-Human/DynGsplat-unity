from plyfile import PlyData
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from tqdm import tqdm
import torch
from dataclasses import dataclass
from torch import nn
from torch_scatter import scatter
from typing import Tuple, Optional
from tqdm import trange
import copy
import gc

from weighted_distance._C import weightedDistance
from torch.optim import Adam
import struct
import argparse
import json

np.set_printoptions(threshold=np.inf) 
class VectorQuantize(nn.Module):
    def __init__(
        self,
        channels: int,
        codebook_size: int = 2**12,
        decay: float = 0.5,
    ) -> None:
        super().__init__()
        self.decay = decay
        self.codebook = nn.Parameter(
            torch.empty(codebook_size, channels), requires_grad=False
        )
        nn.init.kaiming_uniform_(self.codebook)
        self.entry_importance = nn.Parameter(
            torch.zeros(codebook_size), requires_grad=False
        )
        self.eps = 1e-5

    def uniform_init(self, x: torch.Tensor):
        amin, amax = x.aminmax()
        self.codebook.data = torch.rand_like(self.codebook) * (amax - amin) + amin

    def update(self, x: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            min_dists, idx = weightedDistance(x.detach(), self.codebook.detach())
            acc_importance = scatter(
                importance, idx, 0, reduce="sum", dim_size=self.codebook.shape[0]
            )

            ema_inplace(self.entry_importance, acc_importance, self.decay)

            codebook = scatter(
                x * importance[:, None],
                idx,
                0,
                reduce="sum",
                dim_size=self.codebook.shape[0],
            )

            ema_inplace(
                self.codebook,
                codebook / (acc_importance[:, None] + self.eps),
                self.decay,
            )

            return min_dists

    def forward(
        self,
        x: torch.Tensor,
        return_dists: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        min_dists, idx = weightedDistance(x.detach(), self.codebook.detach())
        if return_dists:
            return self.codebook[idx], idx, min_dists
        else:
            return self.codebook[idx], idx


def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def vq_features(
    features: torch.Tensor,
    importance: torch.Tensor,
    codebook_size: int,
    vq_chunk: int = 2**16,
    steps: int = 1000,
    decay: float = 0.8,
    scale_normalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    importance_n = importance/importance.max()
    vq_model = VectorQuantize(
        channels=features.shape[-1],
        codebook_size=codebook_size,
        decay=decay,
    ).to(device=features.device)
    vq_model.uniform_init(features)

    errors = []
    for i in trange(steps):
        batch = torch.randint(low=0, high=features.shape[0], size=[vq_chunk])
        vq_feature = features[batch]
        error = vq_model.update(vq_feature, importance=importance_n[batch]).mean().item()
        errors.append(error)
        if scale_normalize:
            # this computes the trace of the codebook covariance matrices
            # we devide by the trace to ensure that matrices have normalized eigenvalues / scales
            tr = vq_model.codebook[:, [0, 3, 5]].sum(-1)
            vq_model.codebook /= tr[:, None]

    gc.collect()
    torch.cuda.empty_cache()

    _, vq_indices = vq_model(features)
    torch.cuda.synchronize(device=vq_indices.device)
    return vq_model.codebook.data.detach(), vq_indices.detach()

@dataclass
class CompressionSettings:
    """Configuration for color/SH codebook training."""
    codebook_size: int
    importance_prune: float
    importance_include: float
    steps: int
    decay: float
    batch_size: int
def join_features(
    all_features: torch.Tensor,
    keep_mask: torch.Tensor,
    codebook: torch.Tensor,
    codebook_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    keep_features = all_features[keep_mask]
    compressed_features = torch.cat([codebook, keep_features], 0)

    indices = torch.zeros(
        len(all_features), dtype=torch.long, device=all_features.device
    )
    indices[~keep_mask] = codebook_indices
    indices[keep_mask] = torch.arange(len(keep_features), device=indices.device) + len(
        codebook
    )

    return compressed_features, indices

def compress_color(
    gaussians,
    color_importance,
    color_comp,
    color_compress_non_dir,
    path
):
    color_codebook, color_vq_indices = vq_features(
        gaussians,
        color_importance,
        color_comp.codebook_size,
        color_comp.batch_size,
        color_comp.steps,
    )
    return color_codebook.cpu().numpy(), color_vq_indices.detach().contiguous().cpu().int().numpy()

def optimize_codebook(data, initial_indices, initial_codebook, num_iterations=1000):
    if isinstance(initial_codebook, torch.Tensor):
        initial_codebook = initial_codebook.detach().cpu().numpy() 
    codebook = torch.tensor(initial_codebook, dtype=torch.float32, device='cuda', requires_grad=True)
    optimizer = Adam([codebook], lr=0.01)
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        reconstructed = codebook[initial_indices].reshape(-1)
        loss = torch.sum((data - reconstructed)**2)
        loss.backward()
        optimizer.step()
    return codebook.detach().cpu().numpy()

def read_ply_and_export_matrix(file_path):
    plydata = PlyData.read(file_path)
    num_vertices = len(plydata.elements[0])
    num_attributes = len(plydata.elements[0].properties)
    data_matrix = np.zeros((num_vertices, num_attributes))

    for i, attribute in enumerate(plydata.elements[0].properties):
        attribute_name = attribute.name
        attribute_data = np.asarray(plydata.elements[0][attribute_name])
        data_matrix[:, i] = attribute_data

    return data_matrix

def decompress_data(index_value_pairs, data):
    # Initialize an empty numpy array with the original shape
    for index, value in index_value_pairs:
        data[index] = value
    return data

def compress_data(data1, data2):
    difference = data1 - data2
    zero_elements = (difference != 0)
    zero_count = np.sum(zero_elements)
    total_elements = difference.size
    zero_ratio = zero_count / total_elements
    # Find the indices where data1 and data2 differ
    diff_indices = np.where(data1 != data2)[0]
    # Create pairs of (index, new_value)
    index_value_pairs = [[int(index), int(data2[index])] for index in diff_indices]
    return index_value_pairs

def write_ply_rot(vertex_matrix,output_file):
    n, k = vertex_matrix.shape

    attribute_names = ['x', 'y', 'z']
    for i in range(4):
        attribute_names.append('rot_' + str(i))
    for i in range(3):
        attribute_names.append('scale_' + str(i))
    attribute_names.append('opacity')
    
    assert k == len(attribute_names)
    
    n, k = vertex_matrix.shape
   
    with open(output_file, 'wb') as ply_file:
        ply_file.write(b"ply\n")
        ply_file.write(b"format binary_little_endian 1.0\n")
        ply_file.write(b"element vertex %d\n" % n)
        
        for attribute_name in attribute_names:
            ply_file.write(b"property float %s\n" % attribute_name.encode())
        
        ply_file.write(b"end_header\n")
        for i in range(n):
            rot = vertex_matrix[i,3:7]
            vertex_matrix[i,3:7] = rot / np.linalg.norm(rot)
            vertex_data = vertex_matrix[i].astype(np.float32).tobytes()
            ply_file.write(vertex_data)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def write_ply_with_attributes(vertex_matrix, output_file):
    n, k = vertex_matrix.shape

    attribute_names = ['x', 'y', 'z']
    attribute_names.append('nx')
    attribute_names.append('ny')
    attribute_names.append('nz')
    for i in range(3):
        attribute_names.append('f_dc_' + str(i))
    for i in range(45):
        attribute_names.append('f_rest_' + str(i))
    attribute_names.append('opacity')
    for i in range(3):
        attribute_names.append('scale_' + str(i))
    
    for i in range(4):
        attribute_names.append('rot_' + str(i))

    assert k == len(attribute_names)
    
    n, k = vertex_matrix.shape
    with open(output_file, 'wb') as ply_file:
        ply_file.write(b"ply\n")
        ply_file.write(b"format binary_little_endian 1.0\n")
        ply_file.write(b"element vertex %d\n" % n)
        
        for attribute_name in attribute_names:
            ply_file.write(b"property float %s\n" % attribute_name.encode())
        
        ply_file.write(b"end_header\n")
        
        for i in range(n):
            rot = vertex_matrix[i,-4:]
            vertex_matrix[i,-4:] = rot / np.linalg.norm(rot)
            vertex_data = vertex_matrix[i].astype(np.float32).tobytes()
            ply_file.write(vertex_data)


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="")
parser.add_argument("--codebook_size", type=int, default=2**14+1)
parser.add_argument("--output_path", type=str, default="./output")
parser.add_argument("--st", type=int, default=0)
parser.add_argument("--ed", type=int, default=100)
parser.add_argument("--len_block", type=int, default=20)
parser.add_argument("--data_name", type=str, default="Bass")


args = parser.parse_args()

data_path = args.data_path

fixed_codebook_size =  args.codebook_size

save_path=os.path.join(args.output_path,"Data")

os.makedirs(save_path, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
start_all=args.st
end_all=args.ed
len_block=args.len_block
num_block=(end_all-start_all+1+len_block-1)//len_block

dgs_dict ={}
dgs_dict["fps"] = 30
dgs_dict["frame_count"] = end_all - start_all + 1
dgs_dict["block_size"] = len_block
dgs_dict["data_path"] = "Data"
dgs_dict["ply_offset"] = start_all

with open(os.path.join(args.output_path,f"{args.data_name}.dgs"), 'w') as f:
    json.dump(dgs_dict, f, ensure_ascii=False, indent=2)
    
for seg in range(0,num_block):
    print(seg ,"in",num_block)
    start = seg*len_block+start_all
    end = min(((seg+1)*len_block+start_all),end_all+1)

    pointcloud_list_rgb = []
    pointcloud_list_sh_1 = []
    pointcloud_list_sh_2 = []
    pointcloud_list_sh_3 = []
    pointcloud_list_os = []
    name_list = ["rgb", "sh"]
    decode_indices = {}
    split_indice_list = {}
    residual_indices = {}
    save_dict = {}

    for i in tqdm(range(start,end)):
        ply_data = os.path.join(data_path,"point_cloud_{}.ply".format(i))
        data = read_ply_and_export_matrix(ply_data)

        data_sh=copy.deepcopy(data[...,6:54])
        sh=copy.deepcopy(data[...,6:54])
        SH_N=16
        for j in range(1,SH_N):
            sh[...,j*3+0]=data_sh[...,(j-1)+3]
            sh[...,j*3+1]=data_sh[...,(j-1)+SH_N+2]
            sh[...,j*3+2]=data_sh[...,(j-1)+2*SH_N+1]
        
        data_rgb = sh[...,0:3]
        data_sh_1 = sh[...,3:12]
        data_sh_2 = sh[...,12:27]
        data_sh_3 = sh[...,27:48]
        data_os = copy.deepcopy(data[...,[55,56,57,54]])
        
        tmp = data[...,[0,1,2,58,59,60,61]]
        data_os[..., -1] = sigmoid(data_os[..., -1])
        data_os[..., 0:3] = np.exp(data_os[..., 0:3])
        tmp= np.concatenate((tmp, data_os), axis=-1)
    
        write_ply_rot(tmp,os.path.join(save_path,'point_cloud_%d' % i))



        pointcloud_list_rgb.append(data_rgb)
        pointcloud_list_sh_1.append(data_sh_1)
        pointcloud_list_sh_2.append(data_sh_2)
        pointcloud_list_sh_3.append(data_sh_3)
        pointcloud_list_os.append(data_os)
    
    print(data[0].size)
    concatenated_data_rgb = np.concatenate(pointcloud_list_rgb, axis=0)
    concatenated_data_sh_1 = np.concatenate(pointcloud_list_sh_1, axis=0)
    concatenated_data_sh_2 = np.concatenate(pointcloud_list_sh_2, axis=0)
    concatenated_data_sh_3 = np.concatenate(pointcloud_list_sh_3, axis=0)

    concatenated_data_tensor_rgb = torch.from_numpy(concatenated_data_rgb).to(dtype=torch.float32, device='cuda')
    concatenated_data_tensor_sh_1 = torch.from_numpy(concatenated_data_sh_1).to(dtype=torch.float32, device='cuda')
    concatenated_data_tensor_sh_2 = torch.from_numpy(concatenated_data_sh_2).to(dtype=torch.float32, device='cuda')
    concatenated_data_tensor_sh_3 = torch.from_numpy(concatenated_data_sh_3).to(dtype=torch.float32, device='cuda')
    print(concatenated_data_tensor_sh_2.shape)
    # rgb and sh
    color_compression_settings = CompressionSettings(
        codebook_size = fixed_codebook_size,
        importance_prune = 0.0,
        importance_include = 0.6*1e-6,
        steps=int(300),
        decay=0.95,
        batch_size=2**18
    )

    rgb_compression_settings = CompressionSettings(
        codebook_size = fixed_codebook_size,
        importance_prune = 0.0,
        importance_include = 0.6*1e-6,
        steps=int(300),
        decay=0.95,
        batch_size=2**18
    )

    color_importance = torch.ones(concatenated_data_tensor_rgb.shape[0]).to(dtype=torch.float32, device='cuda') # weights
    print("Load FINISH!")
    codebook_color,indices_color = compress_color(
        concatenated_data_tensor_rgb,
        color_importance,
        color_compression_settings,
        True,
        "color_color.npz"
    )
    error = np.abs(codebook_color[indices_color] - concatenated_data_tensor_rgb.cpu().numpy())
    save_dict["{}_codebook".format("rgb")] = codebook_color
    save_dict["{}_color_indices".format("rgb")] = indices_color

    codebook_color,indices_color = compress_color(
        concatenated_data_tensor_sh_1,
        color_importance,
        color_compression_settings,
        True,
        "color_sh.npz"
    )
    error = np.abs(codebook_color[indices_color] - concatenated_data_tensor_sh_1.cpu().numpy())

    save_dict["{}_codebook".format("sh_1")] = codebook_color
    save_dict["{}_color_indices".format("sh_1")] = indices_color
    codebook_color,indices_color = compress_color(
        concatenated_data_tensor_sh_2,
        color_importance,
        color_compression_settings,
        True,
        "color_sh.npz"
    )
    error = np.abs(codebook_color[indices_color] - concatenated_data_tensor_sh_2.cpu().numpy())
    save_dict["{}_codebook".format("sh_2")] = codebook_color
    save_dict["{}_color_indices".format("sh_2")] = indices_color

    codebook_color,indices_color = compress_color(
        concatenated_data_tensor_sh_3,
        color_importance,
        color_compression_settings,
        True,
        "color_sh.npz"
    )
    error = np.abs(codebook_color[indices_color] - concatenated_data_tensor_sh_3.cpu().numpy())
    save_dict["{}_codebook".format("sh_3")] = codebook_color
    save_dict["{}_color_indices".format("sh_3")] = indices_color

    # rgb and sh
    rgb_codebook = save_dict["{}_codebook".format("rgb")]
    sh_1_codebook = save_dict["{}_codebook".format("sh_1")]
    sh_2_codebook = save_dict["{}_codebook".format("sh_2")]
    sh_3_codebook = save_dict["{}_codebook".format("sh_3")]

    rgb_indices = save_dict["{}_color_indices".format("rgb")]
    sh_1_indices = save_dict["{}_color_indices".format("sh_1")]
    sh_2_indices = save_dict["{}_color_indices".format("sh_2")]
    sh_3_indices = save_dict["{}_color_indices".format("sh_3")]

    split_indice_list["rgb"] = []
    split_indice_list["sh_1"] = []
    split_indice_list["sh_2"] = []
    split_indice_list["sh_3"] = []

    for time_index in range(start,end):
        point_per_time = int(concatenated_data_rgb.shape[0]/(end-start))
        start_index = (time_index-start) * point_per_time
        end_index = start_index + point_per_time if time_index != end-1 else None
        split_indices = rgb_indices[start_index:end_index]
        split_indice_list["rgb"].append(split_indices)


    for time_index in range(start,end):
        point_per_time = int(concatenated_data_tensor_sh_1.shape[0]/(end-start))
        start_index = (time_index-start)*point_per_time
        end_index = start_index + point_per_time if time_index != end-1 else None
        split_indices = sh_1_indices[start_index:end_index]
        split_indice_list["sh_1"].append(split_indices)
    

    for time_index in range(start,end):
        point_per_time = int(concatenated_data_tensor_sh_2.shape[0]/(end-start))
        start_index = (time_index-start)*point_per_time
        end_index = start_index + point_per_time if time_index != end-1 else None
        split_indices = sh_2_indices[start_index:end_index]
        split_indice_list["sh_2"].append(split_indices)
        

    for time_index in range(start,end):
        point_per_time = int(concatenated_data_tensor_sh_3.shape[0]/(end-start))
        start_index = (time_index-start)*point_per_time
        end_index = start_index + point_per_time if time_index != end-1 else None
        split_indices = sh_3_indices[start_index:end_index]
        split_indice_list["sh_3"].append(split_indices)



    canoical_index_rgb = copy.deepcopy(split_indice_list["rgb"][0])
    canoical_index_sh_1 = copy.deepcopy(split_indice_list["sh_1"][0])
    canoical_index_sh_2 = copy.deepcopy(split_indice_list["sh_2"][0])
    canoical_index_sh_3 = copy.deepcopy(split_indice_list["sh_3"][0])

    residual_indices["rgb"] = []
    residual_indices["sh_1"] = []
    residual_indices["sh_2"] = []
    residual_indices["sh_3"] = []

    for time_index in range(start+1,end):
        tmp_index_rgb = split_indice_list["rgb"][time_index-start]
        tmp_index_sh_1 = split_indice_list["sh_1"][time_index-start]
        tmp_index_sh_2 = split_indice_list["sh_2"][time_index-start]
        tmp_index_sh_3 = split_indice_list["sh_3"][time_index-start]

        index_value_pairs_rgb = compress_data(canoical_index_rgb, tmp_index_rgb)
        index_value_pairs_sh_1 = compress_data(canoical_index_sh_1, tmp_index_sh_1)
        index_value_pairs_sh_2 = compress_data(canoical_index_sh_2, tmp_index_sh_2)
        index_value_pairs_sh_3 = compress_data(canoical_index_sh_3, tmp_index_sh_3)

        residual_indices["rgb"].append(index_value_pairs_rgb)
        residual_indices["sh_1"].append(index_value_pairs_sh_1)
        residual_indices["sh_2"].append(index_value_pairs_sh_2)
        residual_indices["sh_3"].append(index_value_pairs_sh_3)

        canoical_index_rgb = tmp_index_rgb
        canoical_index_sh_1 = tmp_index_sh_1
        canoical_index_sh_2 = tmp_index_sh_2
        canoical_index_sh_3 = tmp_index_sh_3



    np.array(save_dict["{}_codebook".format("rgb")]).tofile(os.path.join(save_path,"codebook_rgb_{}.bytes".format(seg)))
    np.array(save_dict["{}_codebook".format("sh_1")]).tofile(os.path.join(save_path,"codebook_sh_1_{}.bytes".format(seg)))
    np.array(save_dict["{}_codebook".format("sh_2")]).tofile(os.path.join(save_path,"codebook_sh_2_{}.bytes".format(seg)))
    np.array(save_dict["{}_codebook".format("sh_3")]).tofile(os.path.join(save_path,"codebook_sh_3_{}.bytes".format(seg)))

    print(save_dict["{}_codebook".format("sh_1")].size)
    print(save_dict["{}_codebook".format("sh_2")].size)
    print(save_dict["{}_codebook".format("sh_3")].size)


    canoical_index_dict = []
    canoical_index_dict.append(copy.deepcopy(split_indice_list["rgb"][0]))
    canoical_index_dict.append(copy.deepcopy(split_indice_list["sh_1"][0]))
    canoical_index_dict.append(copy.deepcopy(split_indice_list["sh_2"][0]))
    canoical_index_dict.append(copy.deepcopy(split_indice_list["sh_3"][0]))

    canoical_index_dict=np.array(canoical_index_dict)
    canoical_index_dict=np.transpose(canoical_index_dict)

    canoical_index_dict.tofile(os.path.join(save_path,"canonical_index_{}.bytes".format(seg)))

    index_save_dict=[]
    header=[]
    number=np.uint8(end-start-1)
    print(number)
    offset = number * 4
    
    for time_index in range(start+1,end):
        time_dict=[]
        time_dict.append(np.array(residual_indices["rgb"][time_index-start-1],dtype=np.int32))
        time_dict.append(np.array(residual_indices["sh_1"][time_index-start-1],dtype=np.int32))
        time_dict.append(np.array(residual_indices["sh_2"][time_index-start-1],dtype=np.int32))
        time_dict.append(np.array(residual_indices["sh_3"][time_index-start-1],dtype=np.int32))
        
        index_save_dict.append(time_dict)
        offset_1 = offset + 2*len(residual_indices["rgb"][time_index-start-1])
        offset_2 = offset_1 + 2*len(residual_indices["sh_1"][time_index-start-1])
        offset_3 = offset_2 + 2*len(residual_indices["sh_2"][time_index-start-1])
        offset_4 = offset_3 + 2*len(residual_indices["sh_3"][time_index-start-1])
        header.append([offset_1,offset_2,offset_3,offset_4])
        offset = offset_4

    header=np.array(header,dtype=np.int32)
    with open(os.path.join(save_path,"index_{}.bytes".format(seg)), 'w') as f:
        header.tofile(f)
        for time_dict in index_save_dict:
            for arr in time_dict:
                arr.tofile(f)
    

