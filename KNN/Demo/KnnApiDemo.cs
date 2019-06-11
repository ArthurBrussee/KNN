using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using KNN;
using Unity.Jobs;
using Random = Unity.Mathematics.Random;

public class KnnApiDemo : MonoBehaviour  {
	// Api demo: 
	void Start() {
		const int kNeighbours = 10;

		// First let's create a point cloud
		var points = new NativeArray<float3>(100000, Allocator.Persistent);
		var rand = new Random(123456);
		for (int i = 0; i < points.Length; ++i) {
			points[i] = rand.NextFloat3();
		}
		
		// Create a container that accelerates querying for neighbours
		var knnContainer = new KnnContainer(points);

		// Create an object that allocates memory necessary for queries
		// We can't do this in a job so it is seperated out
		var cache = KnnQueryCache.Create(10);

		// Most basic usage:
		// Get 10 nearest neighbours as indices into our points array!
		// This is NOT burst accelerated yet! Waiting for Unity to implement compiling delegates with Burst...
		var result = new NativeArray<int>(kNeighbours, Allocator.TempJob);
		knnContainer.KNearest(float3.zero, result, cache);
		
		// Get a job to do the query.
		var queryJob = knnContainer.KNearestAsync(new float3(1.0f, 1.0f, 1.0f), result, cache);
		
		// And just run immediatly on the main thread for now. Uses Burst!
		queryJob.Schedule().Complete();

		// Or maybe we want to query neighbours for multiple points.
		const int queryPoints = 1024;
		
		// Keep an array of neighbour indices of all points
		var results = new NativeArray<int>(queryPoints * kNeighbours, Allocator.TempJob);
		var queryPositions = new NativeArray<float3>(queryPoints, Allocator.TempJob);

		for (int i = 0; i < queryPoints; ++i) {
			queryPositions[i] = rand.NextFloat3() * 0.1f;
		}	

		var batchQueryJob = knnContainer.KNearestAsync(queryPositions, results, cache);

		// And just run immediatly on main thread for now
		batchQueryJob.Schedule().Complete();

		// Or maybe we're querying a _ton_ of points (1024 isn't that much but we'll roll with it...)
		// It's a little cumbersome, but we can multi-thread this ourselves
		// If anyone finds a way to expose this as a clean API, go for it
		// But note thar be dragons - tricky to get right! Main trouble is we need a cache per thread...
		const int jobCount = 8;
		var handles = new NativeArray<JobHandle>(jobCount, Allocator.TempJob);
		var caches = new KnnQueryCache[jobCount];

		for (int t = 0; t < jobCount; ++t) {
			int scheduleRange = queryPositions.Length / jobCount;
			int start = scheduleRange * t;
			int end = t == jobCount - 1 ? queryPositions.Length : scheduleRange * (t + 1);

			caches[t] = KnnQueryCache.Create(kNeighbours);
			var posSlice = queryPositions.Slice(start, end - start);
			var resultSlice = results.Slice(start * kNeighbours, (end - start) * kNeighbours);
			var job = knnContainer.KNearestAsync(posSlice, resultSlice, caches[t]);
			handles[t] = job.Schedule();
		}

		// Wait for all jobs to be done
		JobHandle.CompleteAll(handles);

		for (int t = 0; t < jobCount; ++t) {
			caches[t].Dispose();
		}

		// Get rid of memory we're using
		knnContainer.Dispose();
		handles.Dispose();
		queryPositions.Dispose();
		results.Dispose();
		cache.Dispose();
		points.Dispose();
		result.Dispose();
	}
}
