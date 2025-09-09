using UnityEngine;

namespace DynGsplat
{
    public class DynGplatBlockAsset : ScriptableObject
    {
        [HideInInspector] public DynGsplatFrameAsset[] Frames;
        [HideInInspector] public byte[] CanonicalIndex;
        [HideInInspector] public byte[] ResidualIndex;
        [HideInInspector] public byte[] CodebookRGB;
        [HideInInspector] public byte[] CodebookSH1;
        [HideInInspector] public byte[] CodebookSH2;
        [HideInInspector] public byte[] CodebookSH3;
    }
}