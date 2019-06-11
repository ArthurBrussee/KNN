using Unity.Collections;
using Unity.Mathematics;
using KNN;
using Unity.Jobs;
using Random = Unity.Mathematics.Random;

public static class KnnApiDemo  {
	static void Demo() {
		// First let's create a random point cloud
		var points = new NativeArray<float3>(100000, Allocator.Persistent);
		var rand = new Random(123456);
		for (int i = 0; i < points.Length; ++i) {
			points[i] = rand.NextFloat3();
		}

		// Number of neighbours we want to query
		const int kNeighbours = 10;
		float3 queryPosition = float3.zero;
		
		// Create a container that accelerates querying for neighbours
		var knnContainer = new KnnContainer(points, true, Allocator.TempJob);

		// Most basic usage:
		// Get 10 nearest neighbours as indices into our points array!
		// This is NOT burst accelerated yet! Unity need to implement compiling delegates with Burst
		var result = new NativeArray<int>(kNeighbours, Allocator.TempJob);
		knnContainer.KNearest(queryPosition, result);
		
		// The result array at this point contains indices into the points array with the nearest neighbours!
		
		// Get a job to do the query.
		var queryJob = knnContainer.KNearestAsync(queryPosition, result);
		
		// And just run immediatly on the main thread for now. This uses Burst!
		queryJob.Schedule().Complete();

		// Or maybe we want to query neighbours for multiple points.
		const int queryPoints = 1024;
		
		// Keep an array of neighbour indices of all points
		var results = new NativeArray<int>(queryPoints * kNeighbours, Allocator.TempJob);
		
		// Query at a few random points
		var queryPositions = new NativeArray<float3>(queryPoints, Allocator.TempJob);
		for (int i = 0; i < queryPoints; ++i) {
			queryPositions[i] = rand.NextFloat3() * 0.1f;
		}	

		// Fire up job to get results for all points
		var batchQueryJob = knnContainer.KNearestAsync(queryPositions, results);

		// And just run immediatly on main thread for now
		batchQueryJob.Schedule().Complete();
		
		
		// Or maybe we're querying a _ton_ of points (1024 isn't that much but we'll roll with it)
		// It's a little cumbersome, but we can multi-thread this ourselves
		// This will hopefully someday be wrapped in a clean API
		const int jobCount = 8;
		var handles = new NativeArray<JobHandle>(jobCount, Allocator.TempJob);

		for (int t = 0; t < jobCount; ++t) {
			// Figure out indices to schedule
			int scheduleRange = queryPositions.Length / jobCount;
			int start = scheduleRange * t;
			int scheduleCount = t == jobCount - 1 ? queryPositions.Length - start : scheduleRange;

			var posSlice = queryPositions.Slice(start, scheduleCount);
			var resultSlice = results.Slice(start * kNeighbours, scheduleCount * kNeighbours);
			var job = knnContainer.KNearestAsync(posSlice, resultSlice);
			handles[t] = job.Schedule();
		}

		// Wait for all jobs to be done, will run on multiple threads using burst!
		JobHandle.CompleteAll(handles);
		
		// Now the results array contains all the neighbours!

		knnContainer.Dispose();
		handles.Dispose();
		queryPositions.Dispose();
		results.Dispose();
		points.Dispose();
		result.Dispose();
	}
}
