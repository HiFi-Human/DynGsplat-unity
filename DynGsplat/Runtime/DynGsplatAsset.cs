using System;
using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.AddressableAssets;

namespace DynGsplat
{
    public class DynGsplatAsset : ScriptableObject
    {
        public uint SplatCount;
        public byte SHBands => 3;
        public uint FrameCount;
        public float FPS;
        public uint ResidualIndexMaxSize;
        public uint CanonicalIndexSize;
        public uint CodebookRGBSize;
        public uint CodebookSH1Size;
        public uint CodebookSH2Size;
        public uint CodebookSH3Size;
        public uint BlockSize;
        [HideInInspector] public AssetReferenceT<DynGplatBlockAsset>[] Blocks;
        public uint BlockCount => (uint)Math.Ceiling((double)FrameCount / BlockSize);
        public float Duration => FrameCount / FPS;
    }
}