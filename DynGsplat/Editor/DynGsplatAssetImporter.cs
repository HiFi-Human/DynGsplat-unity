using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Gsplat.Editor;
using Unity.Collections;
using UnityEditor;
using UnityEditor.AddressableAssets;
using UnityEditor.AssetImporters;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.AddressableAssets;
using UnityEditor.AddressableAssets.Settings;
using UnityEditor.AddressableAssets.Settings.GroupSchemas;

namespace DynGsplat.Editor
{
    [ScriptedImporter(1, "dgs")]
    public class DynGsplatAssetImporter : ScriptedImporter
    {
        [System.Serializable]
        class ConfigData
        {
            public float fps;
            public int frame_count;
            public int block_size;
            public string data_path;
            public int ply_offset;
        }

        DynGsplatFrameAsset ReadPly(string plyPath)
        {
            var frameAsset = ScriptableObject.CreateInstance<DynGsplatFrameAsset>();

            var bounds = new Bounds();

            using (var fs = new FileStream(plyPath, FileMode.Open, FileAccess.Read))
            {
                // C# arrays and NativeArrays make it hard to have a "byte" array larger than 2GB :/
                if (fs.Length >= 2 * 1024 * 1024 * 1024L)
                    throw new IOException(
                        $"{plyPath} read error: currently files larger than 2GB are not supported");

                GsplatImporter.ReadPlyHeader(fs, out var vertexCount, out var propertyCount);

                frameAsset.SplatCount = vertexCount;
                frameAsset.Positions = new Vector3[vertexCount];
                frameAsset.Opacities = new float[vertexCount];
                frameAsset.Scales = new Vector3[vertexCount];
                frameAsset.Rotations = new Vector4[vertexCount];


                var buffer = new NativeArray<byte>(propertyCount * 4, Allocator.Temp);
                for (uint i = 0; i < vertexCount; i++)
                {
                    var readBytes = fs.Read(buffer);
                    if (readBytes != propertyCount * 4)
                        throw new IOException(
                            $"{plyPath} read error, unexpected end of file, got {readBytes} bytes at vertex {i}");
                    var properties = buffer.Reinterpret<float>(1);
                    frameAsset.Positions[i] = new Vector3(properties[0], properties[1], properties[2]);
                    frameAsset.Rotations[i] = new Vector4(properties[3], properties[4], properties[5], properties[6])
                        .normalized;
                    frameAsset.Scales[i] = new Vector3((properties[7]), (properties[8]),
                        (properties[9]));
                    frameAsset.Opacities[i] = (properties[10]);


                    if (i == 0) bounds = new Bounds(frameAsset.Positions[i], Vector3.zero);
                    else bounds.Encapsulate(frameAsset.Positions[i]);
                }

                buffer.Dispose();
            }

            frameAsset.Bounds = bounds;
            return frameAsset;
        }

        public override void OnImportAsset(AssetImportContext ctx)
        {
            var jsonContent = File.ReadAllText(ctx.assetPath);
            var config = JsonUtility.FromJson<ConfigData>(jsonContent);
            var asset = ScriptableObject.CreateInstance<DynGsplatAsset>();
            asset.FPS = config.fps;
            asset.FrameCount = (uint)config.frame_count;
            var blockCount = (int)asset.BlockCount;
            var dataPath = Path.Combine(Path.GetDirectoryName(ctx.assetPath), config.data_path);

            for (var i = 0; i < blockCount; i++)
            {
                var blockAsset = ScriptableObject.CreateInstance<DynGplatBlockAsset>();

                var canonicalIndexPath = $"{dataPath}/canonical_index_{i}.bytes";
                var residualIndexPath = $"{dataPath}/index_{i}.bytes";
                var codebookRGBPath = $"{dataPath}/codebook_rgb_{i}.bytes";
                var codebookSH1Path = $"{dataPath}/codebook_sh_1_{i}.bytes";
                var codebookSH2Path = $"{dataPath}/codebook_sh_2_{i}.bytes";
                var codebookSH3Path = $"{dataPath}/codebook_sh_3_{i}.bytes";

                blockAsset.name = $"Block{i}";
                blockAsset.CanonicalIndex = AssetDatabase.LoadAssetAtPath<TextAsset>(canonicalIndexPath);
                blockAsset.ResidualIndex = AssetDatabase.LoadAssetAtPath<TextAsset>(residualIndexPath);
                blockAsset.CodebookRGB = AssetDatabase.LoadAssetAtPath<TextAsset>(codebookRGBPath);
                blockAsset.CodebookSH1 = AssetDatabase.LoadAssetAtPath<TextAsset>(codebookSH1Path);
                blockAsset.CodebookSH2 = AssetDatabase.LoadAssetAtPath<TextAsset>(codebookSH2Path);
                blockAsset.CodebookSH3 = AssetDatabase.LoadAssetAtPath<TextAsset>(codebookSH3Path);
                blockAsset.Frames = new DynGsplatFrameAsset[asset.BlockSize];
                asset.ResidualIndexMaxSize =
                    Math.Max((uint)blockAsset.ResidualIndex.dataSize, asset.ResidualIndexMaxSize);
                asset.CanonicalIndexSize = Math.Max((uint)blockAsset.CanonicalIndex.dataSize, asset.CanonicalIndexSize);
                asset.CodebookRGBSize = Math.Max((uint)blockAsset.CodebookRGB.dataSize, asset.CodebookRGBSize);
                asset.CodebookSH1Size = Math.Max((uint)blockAsset.CodebookSH1.dataSize, asset.CodebookSH1Size);
                asset.CodebookSH2Size = Math.Max((uint)blockAsset.CodebookSH2.dataSize, asset.CodebookSH2Size);
                asset.CodebookSH3Size = Math.Max((uint)blockAsset.CodebookSH3.dataSize, asset.CodebookSH3Size);
                for (var j = 0; j < asset.BlockSize; j++)
                {
                    var frameIndex = i * asset.BlockSize + j;
                    var frameAsset = ReadPly($"{dataPath}/point_cloud_{frameIndex + config.ply_offset}");
                    asset.SplatCount = Math.Max(frameAsset.SplatCount, asset.SplatCount);
                    frameAsset.name = $"Frame{frameIndex}";
                    blockAsset.Frames[j] = frameAsset;
                    ctx.AddObjectToAsset(frameAsset.name, frameAsset);

                    EditorUtility.DisplayProgressBar("Importing DynGsplat Asset", $"Reading Files",
                        (frameIndex + 1) / (float)asset.FrameCount);
                }

                ctx.AddObjectToAsset(blockAsset.name, blockAsset);
            }

            ctx.AddObjectToAsset($"DynGsplatAsset", asset);
            ctx.SetMainObject(asset);
        }
    }

    public class DynGsplatAssetPostprocessor : AssetPostprocessor
    {
        static void OnPostprocessAllAssets(string[] importedAssets, string[] deletedAssets, string[] movedAssets,
            string[] movedFromAssetPaths)
        {
            foreach (var assetPath in importedAssets)
                if (assetPath.EndsWith(".dgs"))
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
                group.AddSchema<BundledAssetGroupSchema>();
            }

            var guid = AssetDatabase.AssetPathToGUID(assetPath);
            if (string.IsNullOrEmpty(guid))
                return;

            var entry = settings.CreateOrMoveEntry(guid, group);
            entry.address = assetPath;
        }
    }
}