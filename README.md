# DynGsplat

This repository contains code for compressing 3DGS PLY sequences and a Unity package for rendering and playing the compressed sequences in Unity. The Unity package depends on [wuyize25/gsplat-unity](https://github.com/wuyize25/gsplat-unity), which is also included in the repository as a submodule.

![video](README.assets/video.webp)

## TODO

- [x] RGB & SH compression based on codebook
- [ ] Position compression based on [Draco](https://github.com/google/draco)
- [ ] Video based compression

## Changelog

### v1.1.1 - 2025/10/21

- Lowered the Unity version requirement to 2021.

### v1.1.0 - 2025/09/12

- Added a streaming option that, when enabled, no longer loads all frame data into memory.
- The DGS file format has been changed. The old file version is no longer supported.

### v1.0.0 - 2025/09/10

- The initial release

## Compress

### Setup

Our provided install method is based on Conda package and environment management:

Create a new environment.
```shell
conda create -n compress python=3.9
conda activate compress
```
First install CUDA and PyTorch, our code is evaluated on CUDA 12.1 and PyTorch 2.1.2. Then install the following dependencies:
```shell
cd CompressScripts/
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install weighted_distance/             
pip install -r requirements.txt
```
### Run

We split the PLY sequence into blocks at fixed intervals.
For each block, we learn RGB and SH codebooks via vector quantization and obtain indices.
Using the first frame’s indices as the baseline, later frames store only sparse differences as (index, value) pairs.

Note that our method adopts only part of the compression scheme described in [DualGS Paper Section 4 COMPRESSION](https://arxiv.org/pdf/2409.08353), specifically the module for compressing RGB and SH attributes, which accounts for roughly one-third of the full algorithm.

```shell
python compress.py \
  --data_path <path to ply path> \
  --output_path <path to your output> \
  --st 0 \
  --ed 99 \
  --codebook_size 2**14+1 \
  --len_block 20 \
  --data_name Bass
```

### Core Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| <code style="white-space: nowrap;">--st</code> | int | Start frame number. |
| <code style="white-space: nowrap;">--ed</code> | int | End frame number. |
| <code style="white-space: nowrap;">--len_block</code> | int | Block size. A block contains how many frames. |
| <code style="white-space: nowrap;">--codebook_size</code> | int | Number of entries in the codebook. Must not exceed the index dtype’s max value.|
| <code style="white-space: nowrap;">--data_name</code> | str | Name of the .dgs file |

It will compress the PLY sequences into DGS format, which consists of a `.dgs` metadata file and a `Data` folder.

## Usage of the Unity Package

### Install and Setup

The unity package DynGsplat depends on [wuyize25/gsplat-unity](https://github.com/wuyize25/gsplat-unity), please install and setup the Gsplat package following [Gsplat/README.md](https://github.com/wuyize25/gsplat-unity/blob/main/README.md). Then install the DynGsplat package (DynGsplat/package.json). DynGsplat also depends on Unity's Addressables package. If you haven't installed it before, Addressables will be installed automatically when you install DynGsplat. However, you need to complete the initial configuration for Addressables before using DynGsplat: click `Window > Asset Management > Addressables > Groups`, and then click "Create Addressables Settings" in the window that appears.

### Import Assets

Copy or drag & drop the folder containing the `.dgs` file to any location within your Unity project's `Assets` folder (except for a `Resources` folder). This package will then automatically read the files and import the `.dgs` file as a `Dyn Gsplat Asset` and `.dgsblk` files as `Dyn Gsplat Block Assets`. The imported assets will be automatically added to an Addressables Group named "DynGsplat Assets". This group is automatically configured to be included in the build, so no extra configuration is needed for packaging.

### Add Dyn Gsplat Renderer

Create a new `Game Object` in the scene, then add and configure the `Dyn Gsplat Renderer` component for it. Please note, to ensure that a `Dyn Gsplat Renderer` truly occupies only two blocks of memory when `Streaming` is enabled, you need to make sure the Bundle Mode for the `DynGsplat Assets` Addressables Group is set to "Pack Separately". The new version will automatically set this when creating the `DynGsplat Assets` Addressables Group.

| Property        | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| Asset Ref       | Assign the `Dyn Gsplat Asset` to be played.                  |
| Async Loading   | Toggles asynchronous loading on or off.                      |
| Streaming       | When disabled, all frame data is loaded into memory at runtime; when enabled, it uses at most the memory size of two blocks. |
| Is Playing      | Plays/Pauses the playback.                                   |
| Gamma To Linear | Coverts color space. See [Gsplat/README.md](https://github.com/wuyize25/gsplat-unity/blob/main/README.md). |

## License

This project is released under the [DynGsplat-unity LICENSE](LICENSE.md). It is built upon several other open-source projects:

- [wuyize25/gsplat-unity](https://github.com/wuyize25/gsplat-unity), MIT License (c) 2025 Yize Wu
- [KeKsBoTer/c3dgs](https://github.com/KeKsBoTer/c3dgs)

## Citation

If you find this code useful for your research, please cite us using the following BibTeX entry. 

```bibtex
@article{jiang2024robust,
    title={Robust dual gaussian splatting for immersive human-centric volumetric videos},
    author={Jiang, Yuheng and Shen, Zhehao and Hong, Yu and Guo, Chengcheng and Wu, Yize and Zhang, Yingliang and Yu, Jingyi and Xu, Lan},
    journal={ACM Transactions on Graphics (TOG)},
    volume={43},
    number={6},
    pages={1--15},
    year={2024},
    publisher={ACM New York, NY, USA}
}

@InProceedings{jiang2025reperformer,
    author={Jiang, Yuheng and Shen, Zhehao and Guo, Chengcheng and Hong, Yu and Su, Zhuo and Zhang, Yingliang and Habermann, Marc and Xu, Lan},
    title={RePerformer: Immersive Human-centric Volumetric Videos from Playback to Photoreal Reperformance},
    booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month={June},
    year={2025},
    pages={11349-11360}
}

@inproceedings{hong2025beam,
    author = {Hong, Yu and Wu, Yize and Shen, Zhehao and Guo, Chengcheng and Jiang, Yuheng and Zhang, Yingliang and Hu, Qiang and Yu, Jingyi and Xu, Lan},
    title = {BEAM: Bridging Physically-based Rendering and Gaussian Modeling for Relightable Volumetric Video},
    year = {2025},
    isbn = {9798400720352},
    publisher = {Association for Computing Machinery},
    url = {https://doi.org/10.1145/3746027.3755233},
    doi = {10.1145/3746027.3755233},
    booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
    pages = {7968–7977},
    numpages = {10},
    series = {MM '25}
}

@inproceedings{jiang2025topology,
    author = {Jiang, Yuheng and Guo, Chengcheng and Wu, Yize and Hong, Yu and Zhu, Shengkun and Shen, Zhehao and Zhang, Yingliang and Jiao, Shaohui and Su, Zhuo and Xu, Lan and Habermann, Marc and Theobalt, Christian},
    title = {Topology-Aware Optimization of Gaussian Primitives for Human-Centric Volumetric Videos},
    year = {2025},
    isbn = {9798400721373},
    publisher = {Association for Computing Machinery},
    url = {https://doi.org/10.1145/3757377.3763975},
    doi = {10.1145/3757377.3763975},
    booktitle = {Proceedings of the SIGGRAPH Asia 2025 Conference Papers},
    articleno = {2},
    numpages = {12},
    series = {SA Conference Papers '25}
}
```

