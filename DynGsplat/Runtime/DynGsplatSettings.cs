using System.IO;
using Gsplat;
using UnityEditor;
using UnityEngine;

namespace DynGsplat
{
    public class DynGsplatSettings : ScriptableObject
    {
        const string k_dynGsplatSettingsResourcesPath = "DynGsplatSettings";

        const string k_dynGsplatSettingsPath =
            "Assets/DynGsplat/Settings/Resources/" + k_dynGsplatSettingsResourcesPath + ".asset";

        static DynGsplatSettings s_instance;

        public static DynGsplatSettings Instance
        {
            get
            {
                if (s_instance)
                    return s_instance;

                var settings = Resources.Load<DynGsplatSettings>(k_dynGsplatSettingsResourcesPath);
#if UNITY_EDITOR
                if (!settings)
                {
                    var assetPath = Path.GetDirectoryName(k_dynGsplatSettingsPath);
                    if (!Directory.Exists(assetPath))
                        Directory.CreateDirectory(assetPath);

                    settings = CreateInstance<DynGsplatSettings>();
                    settings.ComputeShader = AssetDatabase.LoadAssetAtPath<ComputeShader>(
                        "Packages/org.hifihuman.dyngsplat/Runtime/DynGsplatDecoder.compute");
                    AssetDatabase.CreateAsset(settings, k_dynGsplatSettingsPath);
                    AssetDatabase.SaveAssets();
                }
#endif

                s_instance = settings;
                return s_instance;
            }
        }

        public ComputeShader ComputeShader;
    }
}