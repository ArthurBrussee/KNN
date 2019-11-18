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
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace KNN.Internal {
	public unsafe struct MinHeap : IDisposable {
		[NativeDisableContainerSafetyRestriction]
		float* heap;

		[NativeDisableContainerSafetyRestriction]
		KdQueryNode* objs;

		public int Count;
		Allocator m_allocator;
		
		public MinHeap(int maxDepth, Allocator allocator) {
			objs = UnsafeUtilityEx.AllocArray<KdQueryNode>(maxDepth + 1, allocator);
			heap = UnsafeUtilityEx.AllocArray<float>(maxDepth + 1, allocator);
			Count = 0;
			m_allocator = allocator;
		}

		static int Parent(int index) {
			return index >> 1;
		}

		static int Left(int index) {
			return index << 1;
		}

		static int Right(int index) {
			return (index << 1) | 1;
		}

		void BubbleDownMin(int index) {
			int l = Left(index);
			int r = Right(index);

			// bubbling down, 2 kids
			while (r <= Count) {
				// if heap property is violated between index and Left child
				if (heap[index] > heap[l]) {
					if (heap[l] > heap[r]) {
						Swap(index, r); // right has smaller priority
						index = r;
					} else {
						Swap(index, l); // left has smaller priority
						index = l;
					}
				} else {
					// if heap property is violated between index and R
					if (heap[index] > heap[r]) {
						Swap(index, r);
						index = r;
					} else {
						index = l;
						l = Left(index);
						break;
					}
				}

				l = Left(index);
				r = Right(index);
			}

			// only left & last children available to test and swap
			if (l <= Count && heap[index] > heap[l]) {
				Swap(index, l);
			}
		}

		void BubbleUpMin(int index) {
			int p = Parent(index);

			//swap, until Heap property isn't violated anymore
			while (p > 0 && heap[p] > heap[index]) {
				Swap(p, index);
				index = p;
				p = Parent(index);
			}
		}

		void Swap(int a, int b) {
			float tempHeap = heap[a];
			KdQueryNode tempObjs = objs[a];

			heap[a] = heap[b];
			objs[a] = objs[b];

			heap[b] = tempHeap;
			objs[b] = tempObjs;
		}

		public void PushObj(KdQueryNode obj, float h) {
			Count++;
			heap[Count] = h;
			objs[Count] = obj;
			BubbleUpMin(Count);
		}

		public KdQueryNode PopObj() {
			KdQueryNode result = objs[1];

			heap[1] = heap[Count];
			objs[1] = objs[Count];
			objs[Count] = default;
			Count--;

			if (Count != 0) {
				BubbleDownMin(1);
			}

			return result;
		}

		public void Dispose() {
			UnsafeUtility.Free(heap, m_allocator);
			UnsafeUtility.Free(objs, m_allocator);
		}
	}
}