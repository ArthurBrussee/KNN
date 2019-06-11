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

using Unity.Collections;

namespace KNN.Internal {
	// array start at index 1
	public struct KSmallestHeap {
		NativeArray<int> objs; //objects
		NativeArray<float> heap;

		public int Count;
		int maxSize;

		public bool Full => maxSize == Count;
		public float HeadValue => heap[1];
		
		public KSmallestHeap(int maxEntries, Allocator allocator) {
			maxSize = maxEntries;
			heap = new NativeArray<float>(maxEntries + 1, allocator);
			objs = new NativeArray<int>(maxEntries + 1, allocator);
			Count = 0;
		}
		
		public void Clear() {
			Count = 0;
		}

		int Parent(int index) {
			return index >> 1;
		}

		int Left(int index) {
			return index << 1;
		}

		int Right(int index) {
			return (index << 1) | 1;
		}

		// bubble down, MaxHeap version
		void BubbleDownMax(int index) {
			int l = Left(index);
			int r = Right(index);

			// bubbling down, 2 kids
			while (r <= Count) {
				// if heap property is violated between index and Left child
				if (heap[index] < heap[l]) {
					if (heap[l] < heap[r]) {
						Swap(index, r); // left has bigger priority
						index = r;
					} else {
						Swap(index, l); // right has bigger priority
						index = l;
					}
				} else {
					// if heap property is violated between index and R
					if (heap[index] < heap[r]) {
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
			if (l <= Count && heap[index] < heap[l]) {
				Swap(index, l);
			}
		}

		// bubble up, MaxHeap version
		void BubbleUpMax(int index) {
			int p = Parent(index);

			//swap, until Heap property isn't violated anymore
			while (p > 0 && heap[p] < heap[index]) {
				Swap(p, index);

				index = p;
				p = Parent(index);
			}
		}

		// bubble down, MinHeap version
		void Swap(int a, int b) {
			float tempHeap = heap[a];
			int tempObjs = objs[a];

			heap[a] = heap[b];
			objs[a] = objs[b];

			heap[b] = tempHeap;
			objs[b] = tempObjs;
		}

		public void PushObj(int obj, float h) {
			// if heap full
			if (Count == maxSize) {
				// if Heads priority is smaller than input priority, then ignore that item
				if (HeadValue < h) {
				} else {
					heap[1] = h; // remove top element
					objs[1] = obj;
					BubbleDownMax(1); // bubble it down
				}
			} else {
				Count++;
				heap[Count] = h;
				objs[Count] = obj;
				BubbleUpMax(Count);
			}
		}

		public int PopObj() {
			int result = objs[1];
			heap[1] = heap[Count];
			objs[1] = objs[Count];
			Count--;
			BubbleDownMax(1);
			return result;
		}

		public int PopObj(out float heapValue) {
			heapValue = heap[1];
			return PopObj();
		}

		public void Dispose() {
			heap.Dispose();
			objs.Dispose();
		}
	}
}