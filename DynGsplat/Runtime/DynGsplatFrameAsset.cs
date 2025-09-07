using UnityEngine;

namespace DynGsplat
{
    public class DynGsplatFrameAsset : ScriptableObject
    {
        public uint SplatCount;
        public byte SHBands => 3;
        public Bounds Bounds;
        [HideInInspector] public Vector3[] Positions;
        [HideInInspector] public Vector3[] Scales;
        [HideInInspector] public Vector4[] Rotations;
        [HideInInspector] public float[] Opacities;
    }
}