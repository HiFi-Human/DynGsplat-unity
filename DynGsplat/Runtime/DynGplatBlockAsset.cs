using UnityEngine;

namespace DynGsplat
{
    public class DynGplatBlockAsset : ScriptableObject
    {
        [HideInInspector] public DynGsplatFrameAsset[] Frames;
        public TextAsset CanonicalIndex;
        public TextAsset ResidualIndex;
        public TextAsset CodebookRGB;
        public TextAsset CodebookSH1;
        public TextAsset CodebookSH2;
        public TextAsset CodebookSH3;
    }
}