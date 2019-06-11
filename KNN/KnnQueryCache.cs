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
using KNN.Internal;

namespace KNN {
	// TODO: Do we grow some of this dynamically?
	public struct KnnQueryCache : IDisposable {
		public int Count;
		public KSmallestHeap Heap;
		public MinHeap MinHeap;
		public NativeArray<KdQueryNode> QueueArray;

		public void Dispose() {
			Heap.Dispose();
			MinHeap.Dispose();
			QueueArray.Dispose();
		}

		public void Reset() {
			Count = 0;
			Heap.Clear();
			MinHeap.Clear();
		}

		public static KnnQueryCache Create(int maxKQuery) {
			KnnQueryCache s;
			s.Count = 0;
			s.Heap = new KSmallestHeap(maxKQuery);
			s.MinHeap = new MinHeap(maxKQuery * 4);
			s.QueueArray = new NativeArray<KdQueryNode>(512, Allocator.Persistent);
			return s;
		}
	}
}