# DynGsplat

This repository contains code for compressing 3DGS PLY sequences and a Unity package for rendering and playing the compressed sequences in Unity. The Unity package depends on [wuyize25/gsplat-unity](https://github.com/wuyize25/gsplat-unity), which is also included in the repository as a submodule.

## TODO

- [x] RGB & SH compression based on codebook
- [ ] Video based compression

## Compress

```shell
python compress.py \
  --data_path <path to ply path> \
  --output_path <path to your output> \
  --st 0 \
  --ed 99 \
  --codebook_size 2**14+1 \
  --len_block 20
```

### Core Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| <code style="white-space: nowrap;">--st</code> | int | Start frame number. |
| <code style="white-space: nowrap;">--ed</code> | int | End frame number. |
| <code style="white-space: nowrap;">--len_block</code> | int | Block size. A block contains how many frames. |

It will compress the PLY sequences into DGS format, which consists of a `.dgs` metadata file and a `Data` folder.

## Usage of the Unity Package

### Install and Setup

The unity package DynGsplat depends on [wuyize25/gsplat-unity](https://github.com/wuyize25/gsplat-unity), please install and setup the Gsplat package following [Gsplat/README.md](https://github.com/wuyize25/gsplat-unity/blob/main/README.md). Then install the DynGsplat package (DynGsplat/package.json). DynGsplat also depends on Unity's Addressables package. If you haven't installed it before, Addressables will be installed automatically when you install DynGsplat. However, you need to complete the initial configuration for Addressables before using DynGsplat: click `Window > Asset Management > Addressables > Groups`, and then click "Create Addressables Settings" in the window that appears.

### Import Assets

Copy the folder containing the `.dgs` file to any location within your Unity project's `Assets` folder (except for a `Resources` folder). Then, when you open the project, Unity will complete the asset import. If you want to import the asset while the project is already open, first copy the `Data` folder (located at the same level as the `.dgs` file) into the project. Wait for Unity to finish importing the `Data` folder, and then copy the `.dgs` file into the project. This will prevent Unity from importing the `.dgs` file before the `Data` folder is fully imported.

After each `.dgs` asset is successfully imported, it will generate a "Dyn Gsplat Asset". This asset will be automatically added to an Addressables Group named "DynGsplat Assets". This group is automatically configured to be included in the build, so no extra configuration is needed for packaging.

### Add Dyn Gsplat Renderer

Create a new `Game Object` in the scene, then add and configure the `Dyn Gsplat Renderer` component for it. At runtime, all frames will be loaded into memory, so please ensure you have enough available memory.

| Property      | Description                                 |
| ------------- | ------------------------------------------- |
| Asset Ref     | Assign the `Dyn Gsplat Asset` to be played. |
| Async Loading | Toggles asynchronous loading on or off.     |
| Is Playing    | Plays/Pauses the playback.                  |

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
```

