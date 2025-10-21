using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using Gsplat.Editor;
using Unity.Collections;
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;
using Unity.SharpZipLib.Utils;

namespace DynGsplat.Editor
{
    [ScriptedImporter(1, "dgsblk")]
    public class DynGsplatBlockAssetImporter : ScriptedImporter
    {
        public override void OnImportAsset(AssetImportContext ctx)
        {
            var extractPath = Path.Combine(Application.temporaryCachePath,
                $"DynGsplatTemp/{ComputeStringHashSHA256(ctx.assetPath)}");
            if (Directory.Exists(extractPath))
                Directory.Delete(extractPath, true);

            ZipUtility.UncompressFromZip(ctx.assetPath, null, extractPath);

            var blockAsset = ScriptableObject.CreateInstance<DynGplatBlockAsset>();

            var canonicalIndexPath = $"{extractPath}/canonical_index.bytes";
            var residualIndexPath = $"{extractPath}/index.bytes";
            var codebookRGBPath = $"{extractPath}/codebook_rgb.bytes";
            var codebookSH1Path = $"{extractPath}/codebook_sh_1.bytes";
            var codebookSH2Path = $"{extractPath}/codebook_sh_2.bytes";
            var codebookSH3Path = $"{extractPath}/codebook_sh_3.bytes";
            
            blockAsset.name = Path.GetFileNameWithoutExtension(ctx.assetPath);
            blockAsset.CanonicalIndex = File.ReadAllBytes(canonicalIndexPath);
            blockAsset.ResidualIndex = File.ReadAllBytes(residualIndexPath);
            blockAsset.CodebookRGB = File.ReadAllBytes(codebookRGBPath);
            blockAsset.CodebookSH1 = File.ReadAllBytes(codebookSH1Path);
            blockAsset.CodebookSH2 = File.ReadAllBytes(codebookSH2Path);
            blockAsset.CodebookSH3 = File.ReadAllBytes(codebookSH3Path);
            blockAsset.Frames = new DynGsplatFrameAsset[Directory.GetFiles(extractPath).Length - 6];

            for (var j = 0; j < blockAsset.Frames.Length; j++)
            {
                var frameAsset = ReadPly($"{extractPath}/point_cloud_{j}.ply");
                frameAsset.name = $"Frame{j}";
                blockAsset.Frames[j] = frameAsset;
                ctx.AddObjectToAsset(frameAsset.name, frameAsset);
            }

            ctx.AddObjectToAsset(blockAsset.name, blockAsset);
            ctx.SetMainObject(blockAsset);
            
            Directory.Delete(extractPath, true);
        }
        
        /// <summary>
        /// Computes the SHA256 hash of a given string.
        /// </summary>
        /// <param name="inputText">The string to hash.</param>
        /// <returns>The lowercase SHA256 hash string.</returns>
        public static string ComputeStringHashSHA256(string inputText)
        {
            if (string.IsNullOrEmpty(inputText))
                return string.Empty;
            using var sha256 = SHA256.Create();
            var inputBytes = Encoding.UTF8.GetBytes(inputText);
            var hashBytes = sha256.ComputeHash(inputBytes);
            var sb = new StringBuilder();
            foreach (var b in hashBytes)
                sb.Append(b.ToString("x2"));
            return sb.ToString();
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

                var buffer = new byte[propertyCount * sizeof(float)];
                for (uint i = 0; i < vertexCount; i++)
                {
                    var readBytes = fs.Read(buffer);
                    if (readBytes != propertyCount * 4)
                        throw new IOException(
                            $"{plyPath} read error, unexpected end of file, got {readBytes} bytes at vertex {i}");
                    var properties = MemoryMarshal.Cast<byte, float>(buffer);
                    frameAsset.Positions[i] = new Vector3(properties[0], properties[1], properties[2]);
                    frameAsset.Rotations[i] = new Vector4(properties[3], properties[4], properties[5], properties[6])
                        .normalized;
                    frameAsset.Scales[i] = new Vector3(properties[7], properties[8], properties[9]);
                    frameAsset.Opacities[i] = properties[10];

                    if (i == 0) bounds = new Bounds(frameAsset.Positions[i], Vector3.zero);
                    else bounds.Encapsulate(frameAsset.Positions[i]);
                }
            }

            frameAsset.Bounds = bounds;
            return frameAsset;
        }
    }
}