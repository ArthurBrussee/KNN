using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using KNN;
using Unity.Jobs;
using UnityEngine.Experimental.ParticleSystemJobs;

// Ideally you really would use something like ECS to visualize the point cloud
// And query from a job component system
// But this is meant to be a really simple demo and not an ECS demo
// I might look into deeper ECS integration at some point
public class KnnVisualizationDemo : MonoBehaviour {
	public int ParticleCount = 20000;
	public int QueryK = 20;
	
	ParticleSystem m_system;

	NativeArray<float3> m_queryPositions;
	NativeArray<Color32> m_queryColors;
	
	NativeArray<float3> m_points;
	NativeArray<int> m_results;
	
	KnnContainer m_container;
	KnnQueryCache m_cache;

	int frame = 0;
	
	void Start() {
		m_system = GetComponent<ParticleSystem>();

		m_system.Emit(ParticleCount);
		m_points = new NativeArray<float3>(ParticleCount, Allocator.Persistent);

		m_cache = KnnQueryCache.Create(QueryK);
		
		// Create a container that accelerates querying for neighbours
		m_container = new KnnContainer(m_points, build:false); // Skip building for now. We rebuild every frame
	}

	void OnDestroy() {
		m_points.Dispose();
		m_cache.Dispose();
		m_container.Dispose();
		m_results.Dispose();
		m_queryPositions.Dispose();
		m_queryColors.Dispose();
	}
	
	// [BurstCompile(CompileSynchronously = true)]
	struct ParticleJob : IParticleSystemJob {
		[ReadOnly] public NativeArray<int> KnnResults;
		public NativeArray<float3> Points;

		public NativeArray<Color32> Colors;
		public int K;
		
		public void ProcessParticleSystem(ParticleSystemJobData jobData) {
			var colors = jobData.startColors;
			var positions = jobData.positions;
			
			for (int i = 0; i < jobData.count; ++i) {
				Points[i] = new float3(positions.x[i], positions.y[i], positions.z[i]);
			}
			
			for (int i = 0; i < jobData.count; i++) {
				colors[i] = new Color32(0, 0, 0, 255);
			}

			// Set all neighbours to red
			for (int i = 0; i < KnnResults.Length; i++) {
				colors[KnnResults[i]] = Colors[i / K];
			}
		}
	}

	void Update() {
		// Update particles job to do the colors
		m_system.SetJob(new ParticleJob {
			KnnResults = m_results,
			Points = m_points,
			K = QueryK,
			Colors = m_queryColors
		});
	}
	
	// After particle job
	void LateUpdate() {
		if (frame < 5) {
			++frame;
			return;
		}	

		// Rebuild our datastructure
		var rebuild = m_container.RebuildAsync();
		var rebuildHandle = rebuild.Schedule();
		
		// Get all probe positions / colors
		if (!m_queryPositions.IsCreated || m_queryPositions.Length != QueryProbe.All.Count) {
			if (m_queryPositions.IsCreated) {
				m_queryPositions.Dispose();
				m_results.Dispose();
				m_queryColors.Dispose();
			}

			m_queryPositions = new NativeArray<float3>(QueryProbe.All.Count, Allocator.Persistent);
			m_results = new NativeArray<int>(QueryK * QueryProbe.All.Count, Allocator.Persistent);
			m_queryColors = new NativeArray<Color32>(QueryProbe.All.Count, Allocator.Persistent);
		}

		for (int i = 0; i < QueryProbe.All.Count; i++) {
			var p = QueryProbe.All[i];
			m_queryPositions[i] = p.transform.position;
			m_queryColors[i] = p.Color;
		}

		
		// Now do the KNN query
		var query = m_container.KNearestAsync(m_queryPositions, m_results, m_cache);
		
		// Schedule query, dependent on the rebuild
		query.Schedule(rebuildHandle).Complete();
	}
}
