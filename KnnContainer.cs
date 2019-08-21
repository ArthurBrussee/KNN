//MIT License
//
//Copyright(c) 2018 Vili Volčini / viliwonka
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.
//
// Modifed 2019 Arthur Brussee

using System;
using KNN.Internal;
using KNN.Jobs;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace KNN {
	[NativeContainerSupportsDeallocateOnJobCompletion, NativeContainer, System.Diagnostics.DebuggerDisplay("Length = {Points.Length}")]
	public struct KnnContainer : IDisposable {
		// We manage safety by our own sentinel. Disable unity's safety system for internal caches / arrays
		[NativeDisableContainerSafetyRestriction]
		public NativeArray<float3> Points;

		[NativeDisableContainerSafetyRestriction]
		NativeArray<int> m_permutation;

		[NativeDisableContainerSafetyRestriction]
		NativeList<KdNode> m_nodes;

		[NativeDisableContainerSafetyRestriction]
		NativeArray<int> m_rootNodeIndex;

		[NativeDisableContainerSafetyRestriction]
		NativeQueue<int> m_buildQueue;

		KdNode RootNode => m_nodes[m_rootNodeIndex[0]];

#if ENABLE_UNITY_COLLECTIONS_CHECKS
		// Note: MUST be named m_Safey, m_DisposeSentinel exactly
		// ReSharper disable once InconsistentNaming
		internal AtomicSafetyHandle m_Safety;
		[NativeSetClassTypeToNullOnSchedule]
		// ReSharper disable once InconsistentNaming
		internal DisposeSentinel m_DisposeSentinel;
#endif

		const int c_maxPointsPerLeafNode = 128;

		public struct KnnQueryTemp {
			public KSmallestHeap Heap;
			public MinHeap MinHeap;

			public static KnnQueryTemp Create(int k) {
				KnnQueryTemp temp;
				temp.Heap = new KSmallestHeap(k, Allocator.Temp);
				temp.MinHeap = new MinHeap(k * 4, Allocator.Temp);
				return temp;
			}
		}

		public KnnContainer(NativeArray<float3> points, bool buildNow, Allocator allocator) {
			int nodeCountEstimate = 4 * (int) math.ceil(points.Length / (float) c_maxPointsPerLeafNode + 1) + 1;
			Points = points;

			// Both arrays are filled in as we go, so start with uninitialized mem
			m_nodes = new NativeList<KdNode>(nodeCountEstimate, allocator);

			// Dumb way to create an int* essentially..
			m_permutation = new NativeArray<int>(points.Length, allocator, NativeArrayOptions.UninitializedMemory);
			m_rootNodeIndex = new NativeArray<int>(new[] {-1}, allocator);
			m_buildQueue = new NativeQueue<int>(allocator);

#if ENABLE_UNITY_COLLECTIONS_CHECKS
			if (allocator <= Allocator.None) {
				throw new ArgumentException("Allocator must be Temp, TempJob or Persistent", nameof(allocator));
			}

			if (points.Length <= 0) {
				throw new ArgumentOutOfRangeException(nameof(points), "Input points length must be >= 0");
			}

			DisposeSentinel.Create(out m_Safety, out m_DisposeSentinel, 0, allocator);
#endif

			if (buildNow) {
				var rebuild = new KnnRebuildJob(this);
				rebuild.Schedule().Complete();
			}
		}

		// TODO: Really want to make this a burst delegate!
		public void Rebuild() {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
			AtomicSafetyHandle.CheckWriteAndThrow(m_Safety);
#endif

			m_nodes.Clear();

			for (int i = 0; i < m_permutation.Length; ++i) {
				m_permutation[i] = i;
			}

			int rootNode = GetKdNode(MakeBounds(), 0, Points.Length);

			m_rootNodeIndex[0] = rootNode;
			m_buildQueue.Enqueue(rootNode);

			while (m_buildQueue.Count > 0) {
				int index = m_buildQueue.Dequeue();
				SplitNode(index, out int posNodeIndex, out int negNodeIndex);

				if (m_nodes[negNodeIndex].Count > c_maxPointsPerLeafNode) {
					m_buildQueue.Enqueue(posNodeIndex);
				}

				if (m_nodes[posNodeIndex].Count > c_maxPointsPerLeafNode) {
					m_buildQueue.Enqueue(negNodeIndex);
				}
			}
		}

		public void Dispose() {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
			DisposeSentinel.Dispose(ref m_Safety, ref m_DisposeSentinel);
#endif

			m_permutation.Dispose();
			m_nodes.Dispose();
			m_rootNodeIndex.Dispose();
			m_buildQueue.Dispose();
		}

		int GetKdNode(KdNodeBounds bounds, int start, int end) {
			m_nodes.Add(new KdNode {
				Bounds = bounds,
				Start = start,
				End = end,
				PartitionAxis = -1,
				PartitionCoordinate = 0.0f,
				PositiveChildIndex = -1,
				NegativeChildIndex = -1
			});

			return m_nodes.Length - 1;
		}

		/// <summary>
		/// For calculating root node bounds
		/// </summary>
		/// <returns>Boundary of all Vector3 points</returns>
		KdNodeBounds MakeBounds() {
			var max = new float3(float.MinValue, float.MinValue, float.MinValue);
			var min = new float3(float.MaxValue, float.MaxValue, float.MaxValue);
			int even = Points.Length & ~1; // calculate even Length

			// min, max calculations
			// 3n/2 calculations instead of 2n
			for (int i0 = 0; i0 < even; i0 += 2) {
				int i1 = i0 + 1;

				// X Coords
				if (Points[i0].x > Points[i1].x) {
					// i0 is bigger, i1 is smaller
					if (Points[i1].x < min.x) {
						min.x = Points[i1].x;
					}

					if (Points[i0].x > max.x) {
						max.x = Points[i0].x;
					}
				} else {
					// i1 is smaller, i0 is bigger
					if (Points[i0].x < min.x) {
						min.x = Points[i0].x;
					}

					if (Points[i1].x > max.x) {
						max.x = Points[i1].x;
					}
				}

				// Y Coords
				if (Points[i0].y > Points[i1].y) {
					// i0 is bigger, i1 is smaller
					if (Points[i1].y < min.y) {
						min.y = Points[i1].y;
					}

					if (Points[i0].y > max.y) {
						max.y = Points[i0].y;
					}
				} else {
					// i1 is smaller, i0 is bigger
					if (Points[i0].y < min.y) {
						min.y = Points[i0].y;
					}

					if (Points[i1].y > max.y) {
						max.y = Points[i1].y;
					}
				}

				// Z Coords
				if (Points[i0].z > Points[i1].z) {
					// i0 is bigger, i1 is smaller
					if (Points[i1].z < min.z) {
						min.z = Points[i1].z;
					}

					if (Points[i0].z > max.z) {
						max.z = Points[i0].z;
					}
				} else {
					// i1 is smaller, i0 is bigger
					if (Points[i0].z < min.z) {
						min.z = Points[i0].z;
					}

					if (Points[i1].z > max.z) {
						max.z = Points[i1].z;
					}
				}
			}

			// if array was odd, calculate also min/max for the last element
			if (even != Points.Length) {
				// X
				if (min.x > Points[even].x) {
					min.x = Points[even].x;
				}

				if (max.x < Points[even].x) {
					max.x = Points[even].x;
				}

				// Y
				if (min.y > Points[even].y) {
					min.y = Points[even].y;
				}

				if (max.y < Points[even].y) {
					max.y = Points[even].y;
				}

				// Z
				if (min.z > Points[even].z) {
					min.z = Points[even].z;
				}

				if (max.z < Points[even].z) {
					max.z = Points[even].z;
				}
			}

			var b = new KdNodeBounds();
			b.Min = min;
			b.Max = max;
			return b;
		}

		// TODO: When multiple points overlap exactly this function breaks.
		/// <summary>
		/// Recursive splitting procedure
		/// </summary>
		unsafe void SplitNode(int parentIndex, out int posNodeIndex, out int negNodeIndex) {
			KdNode parent = m_nodes[parentIndex];

			// center of bounding box
			KdNodeBounds parentBounds = parent.Bounds;
			float3 parentBoundsSize = parentBounds.Size;

			// Find axis where bounds are largest
			int splitAxis = 0;
			float axisSize = parentBoundsSize.x;

			if (axisSize < parentBoundsSize.y) {
				splitAxis = 1;
				axisSize = parentBoundsSize.y;
			}

			if (axisSize < parentBoundsSize.z) {
				splitAxis = 2;
			}

			// Our axis min-max bounds
			float boundsStart = parentBounds.Min[splitAxis];
			float boundsEnd = parentBounds.Max[splitAxis];

			// Calculate the spiting coords
			float splitPivot = CalculatePivot(parent.Start, parent.End, boundsStart, boundsEnd, splitAxis);

			// 'Spiting' array to two sub arrays
			int splittingIndex = Partition(parent.Start, parent.End, splitPivot, splitAxis);

			// Negative / Left node
			float3 negMax = parentBounds.Max;
			negMax[splitAxis] = splitPivot;

			var bounds = parentBounds;
			bounds.Max = negMax;
			negNodeIndex = GetKdNode(bounds, parent.Start, splittingIndex);

			parent.PartitionAxis = splitAxis;
			parent.PartitionCoordinate = splitPivot;
			
			// Positive / Right node
			float3 posMin = parentBounds.Min;
			posMin[splitAxis] = splitPivot;

			bounds = parentBounds;
			bounds.Min = posMin;
			posNodeIndex = GetKdNode(bounds, splittingIndex, parent.End);

			parent.NegativeChildIndex = negNodeIndex;
			parent.PositiveChildIndex = posNodeIndex;

			// Write back node to array to update those values
			m_nodes[parentIndex] = parent;
		}

		/// <summary>
		/// Sliding midpoint splitting pivot calculation
		/// 1. First splits node to two equal parts (midPoint)
		/// 2. Checks if elements are in both sides of splitted bounds
		/// 3a. If they are, just return midPoint
		/// 3b. If they are not, then points are only on left or right bound.
		/// 4. Move the splitting pivot so that it shrinks part with points completely (calculate min or max dependent) and return.
		/// </summary>
		/// <param name="start"></param>
		/// <param name="end"></param>
		/// <param name="boundsStart"></param>
		/// <param name="boundsEnd"></param>
		/// <param name="axis"></param>
		/// <returns></returns>
		float CalculatePivot(int start, int end, float boundsStart, float boundsEnd, int axis) {
			//! sliding midpoint rule
			float midPoint = (boundsStart + boundsEnd) / 2.0f;

			bool negative = false;
			bool positive = false;

			float negMax = float.MinValue;
			float posMin = float.MaxValue;

			// this for loop section is used both for sorted and unsorted data
			for (int i = start; i < end; i++) {
				float val = Points[m_permutation[i]][axis];

				if (val < midPoint) {
					negative = true;
				} else {
					positive = true;
				}

				if (negative && positive) {
					return midPoint;
				}
			}

			if (negative) {
				for (int i = start; i < end; i++) {
					float val = Points[m_permutation[i]][axis];

					if (negMax < val) {
						negMax = val;
					}
				}

				return negMax;
			}

			for (int i = start; i < end; i++) {
				float val = Points[m_permutation[i]][axis];

				if (posMin > val) {
					posMin = val;
				}
			}

			return posMin;
		}

		/// <summary>
		/// Similar to Hoare partitioning algorithm (used in Quick Sort)
		/// Modification: pivot is not left-most element but is instead argument of function
		/// Calculates splitting index and partially sorts elements (swaps them until they are on correct side - depending on pivot)
		/// Complexity: O(n)
		/// </summary>
		/// <param name="start">Start index</param>
		/// <param name="end">End index</param>
		/// <param name="partitionPivot">Pivot that decides boundary between left and right</param>
		/// <param name="axis">Axis of this pivoting</param>
		/// <returns>
		/// Returns splitting index that subdivides array into 2 smaller arrays
		/// left = [start, pivot),
		/// right = [pivot, end)
		/// </returns>
		int Partition(int start, int end, float partitionPivot, int axis) {
			// note: increasing right pointer is actually decreasing!
			int lp = start - 1; // left pointer (negative side)
			int rp = end; // right pointer (positive side)

			while (true) {
				do {
					// move from left to the right until "out of bounds" value is found
					lp++;
				} while (lp < rp && Points[m_permutation[lp]][axis] < partitionPivot);

				do {
					// move from right to the left until "out of bounds" value found
					rp--;
				} while (lp < rp && Points[m_permutation[rp]][axis] >= partitionPivot);

				if (lp < rp) {
					// swap
					int temp = m_permutation[lp];
					m_permutation[lp] = m_permutation[rp];
					m_permutation[rp] = temp;
				} else {
					return lp;
				}
			}
		}
		
		void PushToHeap(int nodeIndex, float3 tempClosestPoint, float3 queryPosition, ref KnnQueryTemp temp) {
			float sqrDist = math.lengthsq(tempClosestPoint - queryPosition);
			
			KdQueryNode queryNode = new KdQueryNode {
				NodeIndex = nodeIndex,
				TempClosestPoint = tempClosestPoint,
				Distance = sqrDist
			};

			temp.MinHeap.PushObj(queryNode, sqrDist);
		}

		// TODO: really want to make this a burst-compiled delegate!
		public void KNearest(float3 queryPosition, NativeSlice<int> result) {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
			AtomicSafetyHandle.CheckReadAndThrow(m_Safety);
#endif
			var temp = KnnQueryTemp.Create(result.Length);
			KNearest(queryPosition, result, ref temp);
		}
		
		internal unsafe void KNearest(float3 queryPosition, NativeSlice<int> result, ref KnnQueryTemp temp) {
			int k = result.Length;
			temp.Heap.Clear();
			
			var points = Points;
			var permutation = m_permutation;
			var rootNode = RootNode;
			var nodePtr = m_nodes.GetUnsafePtr();
			
			// Biggest Smallest Squared Radius
			float bssr = float.PositiveInfinity;
			float3 rootClosestPoint = rootNode.Bounds.ClosestPoint(queryPosition);
			
			PushToHeap(m_rootNodeIndex[0], rootClosestPoint, queryPosition, ref temp);
			
			// searching
			while (temp.MinHeap.Count > 0) {
				KdQueryNode queryNode = temp.MinHeap.PopObj();

				if (queryNode.Distance > bssr) {
					continue;
				}

				ref KdNode node = ref UnsafeUtilityEx.ArrayElementAsRef<KdNode>(nodePtr, queryNode.NodeIndex);

				if (queryNode.NodeIndex < 0) {
					Debug.LogError("Pushed a corrupt node somehow..." + queryNode.NodeIndex);
					continue;
				}

				if (!node.Leaf) {
					int partitionAxis = node.PartitionAxis;
					float partitionCoord = node.PartitionCoordinate;
					float3 tempClosestPoint = queryNode.TempClosestPoint;

					if (partitionAxis > 2) {
						Debug.Log("Bad partition: " + queryNode.NodeIndex + ", " + partitionAxis);
						continue;
					}

					if (tempClosestPoint[partitionAxis] - partitionCoord < 0) {
						if (node.NegativeChildIndex < 0) {
							Debug.LogError(node.NegativeChildIndex + "," + node.PositiveChildIndex);
						}

						// we already know we are on the side of negative bound/node,
						// so we don't need to test for distance
						// push to stack for later querying
						PushToHeap(node.NegativeChildIndex, tempClosestPoint, queryPosition, ref temp);

						// project the tempClosestPoint to other bound
						tempClosestPoint[partitionAxis] = partitionCoord;

						if (UnsafeUtilityEx.ArrayElementAsRef<KdNode>(nodePtr, queryNode.NodeIndex).Count != 0) {
							PushToHeap(node.PositiveChildIndex, tempClosestPoint, queryPosition, ref temp);
						}
					} else {
						// we already know we are on the side of positive bound/node,
						// so we don't need to test for distance
						// push to stack for later querying
						PushToHeap(node.PositiveChildIndex, tempClosestPoint, queryPosition, ref temp);

						// project the tempClosestPoint to other bound
						tempClosestPoint[partitionAxis] = partitionCoord;

						if (UnsafeUtilityEx.ArrayElementAsRef<KdNode>(nodePtr, queryNode.NodeIndex).Count != 0) {
							PushToHeap(node.NegativeChildIndex, tempClosestPoint, queryPosition, ref temp);
						}
					}
				} else {
					for (int i = node.Start; i < node.End; i++) {
						int index = permutation[i];
						float sqrDist = math.lengthsq(points[index] - queryPosition);

						if (sqrDist <= bssr) {
							temp.Heap.PushObj(index, sqrDist);

							if (temp.Heap.Count == k) {
								bssr = temp.Heap.HeadValue;
							}
						}
					}
				}
			}

			int retCount = result.Length + 1;
			
			for (int i = 1; i < retCount; i++) {
				result[i - 1] = temp.Heap.PopObj();
			}
		}
	}
}