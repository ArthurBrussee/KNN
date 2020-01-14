![Demo Image](http://g2f.nl/0jx82rw)

# KNN

Simple K-Nearest Neighbour library for Unity, using the 'Dots' technology stack (the Burst compiler and Unity's job system). It uses K-D trees to speed up queries.

It is totally free from managed allocations and can run multi-threaded. The Burst compiler heavily vectorizes the searching code.

The implementation is a heavily modified version of a KD-Tree by viliwonka: https://github.com/viliwonka/KDTree . It only includes a sub-set of functionality however.

As a rough benchmark, here are perf numbers on a i7-770hq, querying the k=10 nearest neighbours, 100.000 times, in 100.000 points

![Performance comparison](http://g2f.nl/04y9z7g)

This shows an approximately ~130x speedup over the original implementation! Note that this is very much apples to oranges though (eg. this is multi-threaded, the original implementation was not); I don't mean to pick on the original at all, but does hopefully it shows this implementation is really fast!

# API Overview

```C#	
// First let's create a random point cloud
var points = new NativeArray<float3>(100000, Allocator.Persistent);
var rand = new Random(123456);
for (int i = 0; i < points.Length; ++i) {
    points[i] = rand.NextFloat3();
}

// Number of neighbours we want to query
const int kNeighbours = 10;
float3 queryPosition = float3.zero;

// Create a container that accelerates querying for neighbours.
// The 2nd argument indicates whether we want to build the tree straight away or not
// Let's hold off on building it a little bit
var knnContainer = new KnnContainer(points, false, Allocator.TempJob);

// Whenever your point cloud changes, you can make a job to rebuild the container:
var rebuildJob = new KnnRebuildJob(knnContainer);
rebuildJob.Schedule().Complete();

// Most basic usage:
// Get 10 nearest neighbours as indices into our points array!
// This is NOT burst accelerated yet! Unity need to implement compiling delegates with Burst
var result = new NativeArray<int>(kNeighbours, Allocator.TempJob);
knnContainer.QueryKNearest(queryPosition, result);

// The result array at this point contains indices into the points array with the nearest neighbours!
Profiler.BeginSample("Simple Query");
// Get a job to do the query.
var queryJob = new QueryKNearestJob(knnContainer, queryPosition, result);

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

// Fire up job to get results for all points
var batchQueryJob = new QueryKNearestBatchJob(knnContainer, queryPositions, results);

// And just run immediately now. This will run on multiple threads!
batchQueryJob.ScheduleBatch(queryPositions.Length, queryPositions.Length / 32).Complete();

// Or maybe we're interested in a range around eacht query point
var queryRangeResult = new NativeList<int>(Allocator.TempJob);
var rangeQueryJob = new QueryRangeJob(knnContainer, queryPosition, 2.0f, queryRangeResult);

// Store a list of particles in range
var rangeResults = new NativeArray<RangeQueryResult>(queryPoints, Allocator.TempJob);

// And just run immediately on the main thread for now. This uses Burst!
rangeQueryJob.Schedule().Complete();

// Unfortunately, for batch range queries we do need to decide upfront the maximum nr. of neighbours we allow
// This is due to limitation on allocations within a job.
for (int i = 0; i < rangeResults.Length; ++i) {
    rangeResults[i] = new RangeQueryResult(128, Allocator.TempJob);
}

// Fire up job to get results for all points
var batchRange = new QueryRangeBatchJob(knnContainer, queryPositions, 2.0f, rangeResults);

// And just run immediately now. This will run on multiple threads!
batchRange.ScheduleBatch(queryPositions.Length, queryPositions.Length / 32).Complete();

// Now the results array contains all the neighbours!
queryRangeResult.Dispose();
foreach (var r in rangeResults) {
    r.Dispose();
}
rangeResults.Dispose();
knnContainer.Dispose();
queryPositions.Dispose();
results.Dispose();
points.Dispose();
result.Dispose();
```

# Demo

The demo folder contains 2 demos:

- KnnApiDemo.cs, Illustrates various API usages from a basic to advanced level
- KnnVisualizationDemo.cs: Illustrates a real scene where particles are colored based on their neighbourhood information. Shows how to dynamically rebuild your container and other advanced API usages


## Installation

The project was made as a Unity Package. Just add the git URL to your package manifest and Unity should install it.
