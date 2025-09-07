# DynGsplat

This repository contains code for compressing 3DGS PLY sequences and a Unity package for rendering and playing the compressed sequences in Unity.

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
| <code style="white-space: nowrap;">--len_block</code> | int | An index file contains how many frames.  |
