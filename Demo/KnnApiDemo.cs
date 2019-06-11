using Unity.Collections;
using Unity.Mathematics;
using KNN;
using KNN.Jobs;
using Unity.Jobs;
using UnityEngine.Profiling;
using Random = Unity.Mathematics.Random;

public static class KnnApiDemo  {
	public static void Demo() {
		Profiler.BeginSample("Test Query");
		
		// First let's create a random point cloud
		var points = new NativeArray<float3>(100000, Allocator.Persistent);
		var rand = new Random(123456);
		for (int i = 0; i < points.Length; ++i) {
			points[i] = rand.NextFloat3();
		}

		// Number of neighbours we want to query
		const int kNeighbours = 10;
		float3 queryPosition = float3.zero;
		
		Profiler.BeginSample("Build");
		// Create a container that accelerates querying for neighbours
		var knnContainer = new KnnContainer(points, true, Allocator.TempJob);
		Profiler.EndSample();
		
		
		// Most basic usage:
		// Get 10 nearest neighbours as indices into our points array!
		// This is NOT burst accelerated yet! Unity need to implement compiling delegates with Burst
		var result = new NativeArray<int>(kNeighbours, Allocator.TempJob);
		knnContainer.KNearest(queryPosition, result);

		// The result array at this point contains indices into the points array with the nearest neighbours!
		
		Profiler.BeginSample("Simple Query");
		// Get a job to do the query.
		var queryJob = new KNearestQueryJob(knnContainer, queryPosition, result);
		
		// And just run immediately on the main thread for now. This uses Burst!
		queryJob.Schedule().Complete();
		Profiler.EndSample();

		
		// Or maybe we want to query neighbours for multiple points.
		const int queryPoints = 100000;
		
		// Keep an array of neighbour indices of all points
		var results = new NativeArray<int>(queryPoints * kNeighbours, Allocator.TempJob);
		
		// Query at a few random points
		var queryPositions = new NativeArray<float3>(queryPoints, Allocator.TempJob);
		for (int i = 0; i < queryPoints; ++i) {
			queryPositions[i] = rand.NextFloat3() * 0.1f;
		}	

		Profiler.BeginSample("Batch Query");
		// Fire up job to get results for all points
		var batchQueryJob = new KNearestBatchQueryJob(knnContainer, queryPositions, results);

		// And just run immediately now. This will run on multiple threads!
		batchQueryJob.ScheduleBatch(queryPositions.Length, queryPositions.Length / 32).Complete();
		Profiler.EndSample();
		
		
		// Now the results array contains all the neighbours!
		knnContainer.Dispose();
		queryPositions.Dispose();
		results.Dispose();
		points.Dispose();
		result.Dispose();
		Profiler.EndSample();
	}
}
