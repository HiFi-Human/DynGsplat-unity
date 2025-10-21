using System;
using System.Threading;
using System.Threading.Tasks;
using Gsplat;
using UnityEngine;
using UnityEngine.AddressableAssets;
using UnityEngine.ResourceManagement.AsyncOperations;

namespace DynGsplat
{
    [ExecuteAlways]
    public class DynGsplatRenderer : MonoBehaviour, IGsplat
    {
        struct Block
        {
            public DynGplatBlockAsset Asset;
            public AsyncOperationHandle<DynGplatBlockAsset> Handle;
        }

        public AssetReferenceT<DynGsplatAsset> AssetRef;
        public bool AsyncLoading = true;
        public bool Streaming = true;
        public bool IsPlaying = true;
        public bool GammaToLinear;

        string m_prevAssetGUID;

        ComputeShader ComputeShader => DynGsplatSettings.Instance.ComputeShader;

        DynGsplatAsset m_asset;
        GsplatRendererImpl m_renderer;
        AsyncOperationHandle<DynGsplatAsset> m_assetHandle;
        Block[] m_blocks;
        CancellationTokenSource m_tokenSource;

        public ISorterResource SorterResource => m_renderer.SorterResource;

        GraphicsBuffer m_opacityBuffer;
        GraphicsBuffer m_canonicalIndex;
        GraphicsBuffer m_residualIndex;
        GraphicsBuffer m_codebookColor;
        GraphicsBuffer m_codebookSH1;
        GraphicsBuffer m_codebookSH2;
        GraphicsBuffer m_codebookSH3;

        public uint CurrentFrame { get; private set; }
        float m_currentTime = 0;
        uint m_prevFrame = uint.MaxValue;

        const uint k_slidingWindowSize = 2;
        uint m_slidingWindowIndex = k_slidingWindowSize - 1;
        uint CurrentBlockIndex => CurrentFrame / m_asset.BlockSize;
        uint CurrentLocalFrameIndex => CurrentFrame % m_asset.BlockSize;

        DynGplatBlockAsset CurrentBlock =>
#if UNITY_EDITOR
            m_blocks[Streaming && Application.isPlaying ? m_slidingWindowIndex : CurrentBlockIndex].Asset;
#else
            m_blocks[Streaming ? m_slidingWindowIndex : CurrentBlockIndex].Asset;
#endif

        DynGsplatFrameAsset CurrentFrameAsset => CurrentBlock.Frames[CurrentLocalFrameIndex];

        public bool Valid =>
            AssetRef != null &&
            m_asset &&
            m_blocks != null &&
            CurrentBlock &&
            m_renderer != null;

        public uint SplatCount => m_asset ? m_asset.SplatCount : 0;

        static readonly int k_splatCount = Shader.PropertyToID("_SplatCount");
        static readonly int k_colorBuffer = Shader.PropertyToID("_ColorBuffer");
        static readonly int k_shBuffer = Shader.PropertyToID("_SHBuffer");
        static readonly int k_opacityBuffer = Shader.PropertyToID("_OpacityBuffer");
        static readonly int k_canonicalIndex = Shader.PropertyToID("_CanonicalIndex");
        static readonly int k_residualIndex = Shader.PropertyToID("_ResidualIndex");
        static readonly int k_codebookColor = Shader.PropertyToID("_CodebookColor");
        static readonly int k_codebookSH1 = Shader.PropertyToID("_CodebookSH1");
        static readonly int k_codebookSH2 = Shader.PropertyToID("_CodebookSH2");
        static readonly int k_codebookSH3 = Shader.PropertyToID("_CodebookSH3");
        static readonly int k_localFrame = Shader.PropertyToID("_LocalFrame");
        static readonly int k_blockSize = Shader.PropertyToID("_BlockSize");

        const int k_groupSize = 1024;
        int m_kernelUpdateOpacity = -1;
        int m_kernelLoadBlockData = -1;
        int m_kernelUpdateBlockData = -1;

        async Task LoadAssetAsync(CancellationToken token)
        {
            try
            {
                m_asset = (DynGsplatAsset)AssetRef.Asset;
                if (!m_asset)
                {
                    m_assetHandle = AssetRef.LoadAssetAsync();
                    await m_assetHandle.Task;
                    token.ThrowIfCancellationRequested();

                    if (m_assetHandle.Status != AsyncOperationStatus.Succeeded)
                        throw new Exception("Failed to load asset");

                    m_asset = m_assetHandle.Result;
                }

                var blockCount = Streaming ? Math.Min(k_slidingWindowSize, m_asset.BlockCount) : m_asset.BlockCount;
#if UNITY_EDITOR
                m_blocks = new Block[Application.isPlaying ? blockCount : 1];
#else
                m_blocks = new Block[blockCount];
#endif
                for (var i = 0; i < m_blocks.Length; i++)
                {
                    m_blocks[i].Handle = Addressables.LoadAssetAsync<DynGplatBlockAsset>(m_asset.Blocks[i]);
                    await m_blocks[i].Handle.Task;
                    token.ThrowIfCancellationRequested();
                    if (m_blocks[i].Handle.Status != AsyncOperationStatus.Succeeded)
                        throw new Exception($"Failed to load block {i}");
                    m_blocks[i].Asset = m_blocks[i].Handle.Result;
                }
                
                CreateResourcesForAsset();
                UpdateBuffers();
            }
            catch (OperationCanceledException)
            {
            }
        }

        async Task LoadBlockAsync()
        {
            var targetBlockIndex = (CurrentBlockIndex + k_slidingWindowSize - 1) % m_asset.BlockCount;
            var index = (m_slidingWindowIndex + 1) % k_slidingWindowSize;

            m_blocks[index].Asset = null;
            if (m_blocks[index].Handle.IsValid())
                Addressables.Release(m_blocks[index].Handle);

            var handle = Addressables.LoadAssetAsync<DynGplatBlockAsset>(m_asset.Blocks[targetBlockIndex]);

            await handle.Task;
            if (handle.Status != AsyncOperationStatus.Succeeded)
                throw new Exception($"Failed to load block {targetBlockIndex}");
            m_blocks[index].Asset = handle.Result;
            m_blocks[index].Handle = handle;
        }

        void LoadAssetSync()
        {
            m_asset = (DynGsplatAsset)AssetRef.Asset;
            if (!m_asset)
            {
                m_assetHandle = AssetRef.LoadAssetAsync();
                m_assetHandle.WaitForCompletion();
                if (m_assetHandle.Status != AsyncOperationStatus.Succeeded)
                    throw new Exception("Failed to load asset");
                m_asset = m_assetHandle.Result;
            }

            CreateResourcesForAsset();
            var blockCount = Streaming ? Math.Min(k_slidingWindowSize, m_asset.BlockCount) : m_asset.BlockCount;
#if UNITY_EDITOR
            m_blocks = new Block[Application.isPlaying ? blockCount : 1];
#else
            m_blocks = new Block[blockCount];
#endif
            for (var i = 0; i < m_blocks.Length; i++)
            {
                m_blocks[i].Handle = Addressables.LoadAssetAsync<DynGplatBlockAsset>(m_asset.Blocks[i]);

                m_blocks[i].Handle.WaitForCompletion();
                if (m_blocks[i].Handle.Status != AsyncOperationStatus.Succeeded)
                    throw new Exception($"Failed to load block {i}");
                m_blocks[i].Asset = m_blocks[i].Handle.Result;
            }
        }

        void CreateResourcesForAsset()
        {
            if (!m_asset)
                return;

            m_renderer = new GsplatRendererImpl(m_asset.SplatCount, m_asset.SHBands);

            m_opacityBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)m_asset.SplatCount,
                sizeof(float));
            m_residualIndex = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)m_asset.ResidualIndexMaxSize,
                sizeof(uint));
            m_canonicalIndex = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)m_asset.CanonicalIndexSize,
                sizeof(uint));
            m_codebookColor = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)m_asset.CodebookRGBSize,
                sizeof(uint));
            m_codebookSH1 = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)m_asset.CodebookSH1Size,
                sizeof(uint));
            m_codebookSH2 = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)m_asset.CodebookSH2Size,
                sizeof(uint));
            m_codebookSH3 = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)m_asset.CodebookSH3Size,
                sizeof(uint));
        }

        void DisposeResourcesForAsset()
        {
            m_renderer?.Dispose();
            m_opacityBuffer?.Dispose();
            m_canonicalIndex?.Dispose();
            m_residualIndex?.Dispose();
            m_codebookColor?.Dispose();
            m_codebookSH1?.Dispose();
            m_codebookSH2?.Dispose();
            m_codebookSH3?.Dispose();

            m_renderer = null;
            m_opacityBuffer = null;
            m_canonicalIndex = null;
            m_residualIndex = null;
            m_codebookColor = null;
            m_codebookSH1 = null;
            m_codebookSH2 = null;
            m_codebookSH3 = null;
        }

        void LoadAsset()
        {
            if (AssetRef == null || !AssetRef.RuntimeKeyIsValid())
            {
                m_prevAssetGUID = "";
                return;
            }

            m_prevAssetGUID = AssetRef.AssetGUID;

#if UNITY_EDITOR
            var loadAsync = Application.isPlaying && AsyncLoading;
#else
            var loadAsync = AsyncLoading;
#endif
            if (loadAsync)
            {
                m_tokenSource = new CancellationTokenSource();
                _ = LoadAssetAsync(m_tokenSource.Token);
            }
            else
                LoadAssetSync();
            
            m_currentTime = 0;
            CurrentFrame = 0;
            m_prevFrame = uint.MaxValue;
            m_slidingWindowIndex = k_slidingWindowSize - 1;
        }

        void UnloadAsset()
        {
            DisposeResourcesForAsset();

            if (m_tokenSource != null)
            {
                m_tokenSource.Cancel();
                m_tokenSource.Dispose();
                m_tokenSource = null;
            }

            if (m_blocks != null)
            {
                for (var i = 0; i < m_blocks.Length; i++)
                {
                    var handle = m_blocks[i].Handle;
                    if (handle.IsValid())
                        Addressables.Release(handle);
                }
            }

            if (m_assetHandle.IsValid())
                Addressables.Release(m_assetHandle);
            m_asset = null;
        }

        void OnEnable()
        {
            GsplatSorter.Instance.RegisterGsplat(this);

            m_kernelUpdateOpacity = ComputeShader.FindKernel("UpdateOpacity");
            m_kernelLoadBlockData = ComputeShader.FindKernel("LoadBlockData");
            m_kernelUpdateBlockData = ComputeShader.FindKernel("UpdateBlockData");

            LoadAsset();
        }

        void OnDisable()
        {
            GsplatSorter.Instance.UnregisterGsplat(this);
            UnloadAsset();
        }

        void Update()
        {
            if (AssetRef.AssetGUID != m_prevAssetGUID)
            {
                UnloadAsset();
                LoadAsset();
            }

            if (!Valid || !GsplatSettings.Instance.Valid || !GsplatSorter.Instance.Valid)
                return;

            var prevTime = m_currentTime;

#if UNITY_EDITOR
            if (Application.isPlaying && IsPlaying)
#else
            if (IsPlaying)
#endif
            {
                m_currentTime += Time.deltaTime;

                if (m_currentTime > m_asset.Duration)
                    m_currentTime -= m_asset.Duration;

                CurrentFrame = (uint)(m_currentTime * m_asset.FPS);
            }

            if (CurrentFrame != m_prevFrame)
            {
                if (Streaming && CurrentLocalFrameIndex == 0)
                    m_slidingWindowIndex = (m_slidingWindowIndex + 1) % k_slidingWindowSize;

                if (CurrentBlock)
                {
                    m_prevFrame = CurrentFrame;
                    if (Streaming && CurrentLocalFrameIndex == 0)
                    {
                        _ = LoadBlockAsync();
                    }

                    UpdateBuffers();
                }
                else
                {
                    if (Streaming && CurrentLocalFrameIndex == 0)
                        m_slidingWindowIndex = (m_slidingWindowIndex + k_slidingWindowSize - 1) % k_slidingWindowSize;
                    CurrentFrame = m_prevFrame;
                    m_currentTime = prevTime;
                }
            }

            m_renderer.Render(m_asset.SplatCount, transform, CurrentFrameAsset.Bounds, gameObject.layer, GammaToLinear);
        }

        void UpdateBuffers()
        {
            m_renderer.PositionBuffer.SetData(CurrentFrameAsset.Positions);
            m_renderer.ScaleBuffer.SetData(CurrentFrameAsset.Scales);
            m_renderer.RotationBuffer.SetData(CurrentFrameAsset.Rotations);
            m_opacityBuffer.SetData(CurrentFrameAsset.Opacities);

            ComputeShader.SetInt(k_splatCount, (int)m_asset.SplatCount);
            ComputeShader.SetBuffer(m_kernelUpdateOpacity, k_opacityBuffer, m_opacityBuffer);
            ComputeShader.SetBuffer(m_kernelUpdateOpacity, k_colorBuffer, m_renderer.ColorBuffer);
            ComputeShader.Dispatch(m_kernelUpdateOpacity,
                (int)Math.Ceiling(m_asset.SplatCount / (float)k_groupSize), 1, 1);

            if (CurrentLocalFrameIndex == 0)
            {
                m_residualIndex.SetData(CurrentBlock.ResidualIndex);
                m_canonicalIndex.SetData(CurrentBlock.CanonicalIndex);
                m_codebookColor.SetData(CurrentBlock.CodebookRGB);
                m_codebookSH1.SetData(CurrentBlock.CodebookSH1);
                m_codebookSH2.SetData(CurrentBlock.CodebookSH2);
                m_codebookSH3.SetData(CurrentBlock.CodebookSH3);

                ComputeShader.SetInt(k_splatCount, (int)m_asset.SplatCount);
                ComputeShader.SetBuffer(m_kernelLoadBlockData, k_canonicalIndex, m_canonicalIndex);
                ComputeShader.SetBuffer(m_kernelLoadBlockData, k_codebookColor, m_codebookColor);
                ComputeShader.SetBuffer(m_kernelLoadBlockData, k_codebookSH1, m_codebookSH1);
                ComputeShader.SetBuffer(m_kernelLoadBlockData, k_codebookSH2, m_codebookSH2);
                ComputeShader.SetBuffer(m_kernelLoadBlockData, k_codebookSH3, m_codebookSH3);
                ComputeShader.SetBuffer(m_kernelLoadBlockData, k_colorBuffer, m_renderer.ColorBuffer);
                ComputeShader.SetBuffer(m_kernelLoadBlockData, k_shBuffer, m_renderer.SHBuffer);
                ComputeShader.Dispatch(m_kernelLoadBlockData,
                    (int)Math.Ceiling(m_asset.SplatCount / (float)k_groupSize), 1, 1);
            }
            else
            {
                ComputeShader.SetInt(k_localFrame, (int)CurrentLocalFrameIndex);
                ComputeShader.SetInt(k_blockSize, (int)m_asset.BlockSize);
                ComputeShader.SetBuffer(m_kernelUpdateBlockData, k_residualIndex, m_residualIndex);
                ComputeShader.SetBuffer(m_kernelUpdateBlockData, k_codebookColor, m_codebookColor);
                ComputeShader.SetBuffer(m_kernelUpdateBlockData, k_codebookSH1, m_codebookSH1);
                ComputeShader.SetBuffer(m_kernelUpdateBlockData, k_codebookSH2, m_codebookSH2);
                ComputeShader.SetBuffer(m_kernelUpdateBlockData, k_codebookSH3, m_codebookSH3);
                ComputeShader.SetBuffer(m_kernelUpdateBlockData, k_colorBuffer, m_renderer.ColorBuffer);
                ComputeShader.SetBuffer(m_kernelUpdateBlockData, k_shBuffer, m_renderer.SHBuffer);
                const int groupSizeX = k_groupSize / 4;
                ComputeShader.Dispatch(m_kernelUpdateBlockData,
                    ((int)m_asset.SplatCount + groupSizeX - 1) / groupSizeX, 1, 1);
            }
        }
    }
}