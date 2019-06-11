![Demo Image](http://g2f.nl/0jx82rw)

# KNN

Simple K-Nearest Neighbour library for Unity, using the 'Dots' technology stack (the Burst compiler and Unity's job system). It uses K-D trees to speed up queries.

It is totally free from managed allocations and can run multi-threaded. The Burst compiler heavily vectorizes the searching code.

As a rough benchmark, the included demo rebuilds the KD tree in about ~1.0ms for 20.000 points, and does 10 queries for the 20 nearest neighbours in about ~0.08ms, on a i7-7700hq.

The implementation is a heavily modified version of a KD-Tree by viliwonka: https://github.com/viliwonka/KDTree . It only includes a sub-set of functionality however.

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

// Create a container that accelerates querying for neighbours
var knnContainer = new KnnContainer(points, true, Allocator.TempJob);

// Most basic usage:
// Get 10 nearest neighbours as indices into our points array!
// This is NOT burst accelerated yet! Unity need to implement compiling delegates with Burst
var result = new NativeArray<int>(kNeighbours, Allocator.TempJob);
knnContainer.KNearest(queryPosition, result);

// The result array at this point contains indices into the points array with the nearest neighbours!

// Get a job to do the query.
var queryJob = new KnnQueryJob(knnContainer, queryPosition, result);

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
var batchQueryJob = new KNearestBatchQueryJob(knnContainer, queryPositions, results);

// And just run immediatly now. This will run on multiple threads!
batchQueryJob.ScheduleBatch(queryPositions.Length, 128).Complete();

// Now the results array contains all the neighbours!
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