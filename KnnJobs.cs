using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

// Jobs for querying
namespace KNN.Jobs {
	[BurstCompile(CompileSynchronously = true)]
	public struct KnnQueryJob : IJob {
		[ReadOnly] KnnContainer m_container;
		[WriteOnly] NativeSlice<int> m_result;

		float3 m_queryPosition;

		public KnnQueryJob(KnnContainer container, float3 queryPosition, NativeSlice<int> result) {
			m_result = result;
			m_queryPosition = queryPosition;
			m_container = container;
		}

		void IJob.Execute() {
			m_container.KNearest(m_queryPosition, m_result);
		}
	}

	[BurstCompile(CompileSynchronously = true)]
	public struct KNearestBatchQueryJob : IJobParallelForBatch {
		[ReadOnly] KnnContainer m_container;

		[ReadOnly] 
		NativeSlice<float3> m_queryPositions;

		// Unity really doesn't like it when we write to the same underlying array
		// Even if slices don't overlap... So we're just being dangerous here
		[NativeDisableParallelForRestriction, NativeDisableContainerSafetyRestriction]
		NativeSlice<int> m_results;

		int m_k;

		public KNearestBatchQueryJob(KnnContainer container, NativeArray<float3> queryPositions, NativeSlice<int> results) {
			m_container = container;
			m_queryPositions = queryPositions;
			m_results = results;

		#if ENABLE_UNITY_COLLECTIONS_CHECKS
			if (queryPositions.Length == 0 || results.Length % queryPositions.Length != 0) {
				Debug.LogError("Make sure your results array is a multiple in length of your querypositions array!");
			}
		#endif

			m_k = results.Length / queryPositions.Length;
		}
		
		public void Execute(int startIndex, int count) {
			var temp = KnnContainer.KnnQueryTemp.Create(m_k);
			
			// Write results to proper slice!
			for (int index = startIndex; index < startIndex + count; ++index) {
				var resultsSlice = m_results.Slice(index * m_k, m_k);
				m_container.KNearest(m_queryPositions[index], resultsSlice, temp);
			}
		}
	}

	[BurstCompile(CompileSynchronously = true)]
	public struct KNearestQueryJob : IJob {
		KnnContainer m_container;

		public KNearestQueryJob(KnnContainer container) {
			m_container = container;
		}

		void IJob.Execute() {
			m_container.Rebuild();
		}
	}
}