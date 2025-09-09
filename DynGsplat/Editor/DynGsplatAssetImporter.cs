using System;
using System.IO;
using UnityEditor;
using UnityEditor.AddressableAssets;
using UnityEditor.AssetImporters;
using UnityEngine;
using UnityEngine.AddressableAssets;
using UnityEditor.AddressableAssets.Settings;
using UnityEditor.AddressableAssets.Settings.GroupSchemas;

namespace DynGsplat.Editor
{
    [ScriptedImporter(1, "dgs")]
    public class DynGsplatAssetImporter : ScriptedImporter
    {
        [Serializable]
        class ConfigData
        {
            public string version;
            public float fps;
            public int frame_count;
            public int block_size;
            public string data_path;
            public int ply_offset;
        }

        static string[] GatherDependenciesFromSourceFile(string path)
        {
            var jsonContent = File.ReadAllText(path);
            var config = JsonUtility.FromJson<ConfigData>(jsonContent);
            if (config.version is not "1.1")
                return null;

            var dataPath = Path.Combine(Path.GetDirectoryName(path), config.data_path);
            var blockCount = Mathf.CeilToInt((float)config.frame_count / config.block_size);
            var dependencies = new string[blockCount];
            for (var i = 0; i < blockCount; i++)
                dependencies[i] = Path.Combine(dataPath, $"Block{i}.dgsblk");
            return dependencies;
        }

        public override void OnImportAsset(AssetImportContext ctx)
        {
            var jsonContent = File.ReadAllText(ctx.assetPath);
            var config = JsonUtility.FromJson<ConfigData>(jsonContent);

            if (config.version is not "1.1")
            {
                Debug.LogError("Unsupported DGS file version" + (config.version != null ? $": {config.version}" : ""));
                return;
            }

            var asset = ScriptableObject.CreateInstance<DynGsplatAsset>();
            asset.FPS = config.fps;
            asset.FrameCount = (uint)config.frame_count;
            asset.BlockSize = (uint)config.block_size;
            asset.Blocks = new AssetReferenceT<DynGplatBlockAsset>[asset.BlockCount];
            var blockCount = (int)asset.BlockCount;
            var dataPath = Path.Combine(Path.GetDirectoryName(ctx.assetPath), config.data_path);

            for (var i = 0; i < blockCount; i++)
            {
                var blockPath = Path.Combine(dataPath, $"Block{i}.dgsblk");
                var blockAsset = AssetDatabase.LoadAssetAtPath<DynGplatBlockAsset>(blockPath);
                asset.Blocks[i] = new AssetReferenceT<DynGplatBlockAsset>(AssetDatabase.AssetPathToGUID(blockPath));
                asset.ResidualIndexMaxSize =
                    Math.Max((uint)blockAsset.ResidualIndex.Length, asset.ResidualIndexMaxSize);
                asset.CanonicalIndexSize = Math.Max((uint)blockAsset.CanonicalIndex.Length, asset.CanonicalIndexSize);
                asset.CodebookRGBSize = Math.Max((uint)blockAsset.CodebookRGB.Length, asset.CodebookRGBSize);
                asset.CodebookSH1Size = Math.Max((uint)blockAsset.CodebookSH1.Length, asset.CodebookSH1Size);
                asset.CodebookSH2Size = Math.Max((uint)blockAsset.CodebookSH2.Length, asset.CodebookSH2Size);
                asset.CodebookSH3Size = Math.Max((uint)blockAsset.CodebookSH3.Length, asset.CodebookSH3Size);
                for (var j = 0; j < blockAsset.Frames.Length; j++)
                {
                    var frameIndex = i * asset.BlockSize + j;
                    var frameAsset = blockAsset.Frames[j];
                    asset.SplatCount = Math.Max(frameAsset.SplatCount, asset.SplatCount);
                    frameAsset.name = $"Frame{frameIndex}";
                    blockAsset.Frames[j] = frameAsset;
                    EditorUtility.DisplayProgressBar("Importing DynGsplat Asset", "Reading Files",
                        (frameIndex + 1) / (float)asset.FrameCount);
                }
            }

            ctx.AddObjectToAsset("DynGsplatAsset", asset);
            ctx.SetMainObject(asset);
        }
    }

    public class DynGsplatAssetPostprocessor : AssetPostprocessor
    {
        static void OnPostprocessAllAssets(string[] importedAssets, string[] deletedAssets, string[] movedAssets,
            string[] movedFromAssetPaths)
        {
            foreach (var assetPath in importedAssets)
                if (assetPath.EndsWith(".dgs") || assetPath.EndsWith(".dgsblk"))
                    MakeAssetAddressable(assetPath);
        }

        static void MakeAssetAddressable(string assetPath)
        {
            var settings = AddressableAssetSettingsDefaultObject.Settings;
            if (settings == null)
            {
                Debug.LogError("Addressable Asset Settings not found. Please initialize Addressables first.");
                return;
            }

            var groupName = "DynGsplat Assets";
            var group = settings.FindGroup(groupName);
            if (group == null)
            {
                group = settings.CreateGroup(groupName, false, false, true, null, typeof(AddressableAssetGroupSchema));
                var schema = group.AddSchema<BundledAssetGroupSchema>();
                schema.BundleMode = BundledAssetGroupSchema.BundlePackingMode.PackSeparately;
                schema.Compression = BundledAssetGroupSchema.BundleCompressionMode.Uncompressed;
            }

            var guid = AssetDatabase.AssetPathToGUID(assetPath);
            if (string.IsNullOrEmpty(guid))
                return;

            var entry = settings.CreateOrMoveEntry(guid, group);
            entry.address = assetPath;
        }
    }
}