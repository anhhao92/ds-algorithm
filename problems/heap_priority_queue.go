package problems

import (
	"container/heap"
)

type Point struct {
	dist  int
	index int
}

type Heap[T any] struct {
	values     []T
	comparator func(i, j T) bool
}

func (h Heap[T]) Len() int           { return len(h.values) }
func (h Heap[T]) Less(i, j int) bool { return h.comparator(h.values[i], h.values[j]) }
func (h Heap[T]) Swap(i, j int)      { h.values[i], h.values[j] = h.values[j], h.values[i] }

func (h *Heap[T]) Push(x any) {
	h.values = append(h.values, x.(T))
}

func (h *Heap[T]) Pop() any {
	old := *h
	n := len(old.values)
	x := old.values[n-1]
	h.values = old.values[0 : n-1]
	return x
}

func (h Heap[T]) Peak() T {
	return h.values[0]
}

func kClosest(points [][]int, k int) [][]int {
	minHeap := &Heap[Point]{comparator: func(i, j Point) bool { return i.dist < j.dist }}
	for i, point := range points {
		x, y := point[0], point[1]
		minHeap.values = append(minHeap.values, Point{dist: x*x + y*y, index: i})
	}
	heap.Init(minHeap)
	res := make([][]int, k)
	for i := 0; i < k; i++ {
		p := heap.Pop(minHeap).(Point)
		res[i] = points[p.index]
	}
	return res
}

func partition(nums []int, left int, right int) int {
	pivot := nums[right]
	partitionIndex := left
	for i := left; i < right; i++ {
		if nums[i] < pivot {
			nums[i], nums[partitionIndex] = nums[partitionIndex], nums[i]
			partitionIndex++
		}
	}
	nums[right], nums[partitionIndex] = nums[partitionIndex], nums[right]
	return partitionIndex
}

type IntHeap []int

func (h IntHeap) Len() int           { return len(h) }
func (h IntHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h IntHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *IntHeap) Push(x any) {
	*h = append(*h, x.(int))
}

func (h *IntHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func findKthLargest(nums []int, k int) int {
	// min heap approach
	// h := IntHeap(nums[:k])
	// heap.Init(&h)
	// for i := k; i < len(nums); i++ {
	// 	if nums[i] > h[0] {
	// 		heap.Pop(&h)
	// 		heap.Push(&h, nums[i])
	// 	}
	// }
	// return h[0]
	// QuickSelect approach
	// left, right := 0, len(nums)-1
	// k = len(nums) - k
	// for left <= right {
	// 	newPivotIndex := partition(nums, left, right)
	// 	if newPivotIndex == k {
	// 		return nums[newPivotIndex]
	// 	} else if newPivotIndex > k {
	// 		right = newPivotIndex - 1
	// 	} else {
	// 		left = newPivotIndex + 1
	// 	}
	// }
	// return -1
	k = len(nums) - k
	var quickSelect func(left, right int) int
	quickSelect = func(left, right int) int {
		pivotIndex := left
		pivot := nums[right]
		for i := left; i < right; i++ {
			if nums[i] < pivot {
				nums[i], nums[pivotIndex] = nums[pivotIndex], nums[i]
				pivotIndex++
			}
		}
		//swapping pivot to the final pivot location
		nums[pivotIndex], nums[right] = nums[right], nums[pivotIndex]
		if pivotIndex > k {
			return quickSelect(left, pivotIndex-1)
		}
		if pivotIndex < k {
			return quickSelect(pivotIndex+1, right)
		}
		return nums[pivotIndex]
	}
	return quickSelect(0, len(nums)-1)
}

func LeastInterval(tasks []byte, n int) int {
	freq := map[byte]int{}
	for _, v := range tasks {
		freq[v]--
	}

	taskCounter := []int{}
	for _, v := range freq {
		taskCounter = append(taskCounter, v)
	}
	time := 0
	queue := [][2]int{} // [count, idleTime]
	h := IntHeap(taskCounter)
	heap.Init(&h)
	for h.Len() > 0 || len(queue) > 0 {
		time++
		if h.Len() > 0 {
			count := 1 + heap.Pop(&h).(int)
			if count != 0 {
				queue = append(queue, [2]int{count, n + time})
			}
		}
		if len(queue) > 0 && queue[0][1] == time {
			heap.Push(&h, queue[0][0])
			queue = queue[1:]
		}
	}
	return time
}

/**
 * Your MedianFinder object will be instantiated and called as such:
 * obj := Constructor();
 * obj.AddNum(num);
 * param_2 := obj.FindMedian();
 */
type MedianFinder struct {
	minHeap IntHeap
	maxHeap IntHeap
}

func NewMedianFinder() MedianFinder {
	return MedianFinder{minHeap: IntHeap{}, maxHeap: IntHeap{}}
}

func (this *MedianFinder) AddNum(num int) {
	//[max 1 2 ][4 6 min]
	// add to min heap
	if this.minHeap.Len() == 0 || num > this.minHeap[0] {
		heap.Push(&this.minHeap, num)
	} else {
		heap.Push(&this.maxHeap, -num)
	}
	// balance minHeap+maxHeap
	if this.minHeap.Len() > this.maxHeap.Len()+1 {
		v := heap.Pop(&this.minHeap).(int)
		heap.Push(&this.maxHeap, -v)
	} else if this.minHeap.Len() < this.maxHeap.Len() {
		v := heap.Pop(&this.maxHeap).(int)
		heap.Push(&this.minHeap, -v)
	}
}

func (this *MedianFinder) FindMedian() float64 {
	if this.maxHeap.Len() == this.minHeap.Len() {
		return float64(-1*this.maxHeap[0]+this.minHeap[0]) / 2
	}
	return float64(this.minHeap[0])
}

func MedianSlidingWindow(nums []int, k int) []float64 {
	minHeap := Heap[int]{comparator: func(i, j int) bool { return i < j }}
	maxHeap := Heap[int]{comparator: func(i, j int) bool { return i > j }}
	balanceHeap := func() {
		// balance minHeap and maxHeap
		if minHeap.Len() > maxHeap.Len()+1 {
			v := heap.Pop(&minHeap).(int)
			heap.Push(&maxHeap, v)
		} else if minHeap.Len() < maxHeap.Len() {
			v := heap.Pop(&maxHeap).(int)
			heap.Push(&minHeap, v)
		}
	}
	addNum := func(num int) {
		if minHeap.Len() == 0 || num >= minHeap.Peak() {
			heap.Push(&minHeap, num)
		} else {
			heap.Push(&maxHeap, num)
		}
		balanceHeap()
	}
	removeNum := func(num int) {
		if num >= minHeap.Peak() {
			for i, v := range minHeap.values {
				if v == num {
					heap.Remove(&minHeap, i)
					break
				}
			}
		} else {
			for i, v := range maxHeap.values {
				if v == num {
					heap.Remove(&maxHeap, i)
					break
				}
			}
		}
		balanceHeap()
	}
	findMedian := func() float64 {
		if maxHeap.Len() == minHeap.Len() {
			return float64(maxHeap.Peak()+minHeap.Peak()) / 2.0
		}
		return float64(minHeap.Peak())
	}
	for i := 0; i < k; i++ {
		addNum(nums[i])
	}
	result := []float64{findMedian()}
	for i := k; i < len(nums); i++ {
		removeNum(nums[i-k])
		addNum(nums[i])
		result = append(result, findMedian())
	}

	return result
}

// LC 2583
func kthLargestLevelSum(root *TreeNode, k int) int64 {
	queue := []*TreeNode{root}
	minHeap := &IntHeap{}
	for len(queue) > 0 {
		sum := 0
		for _, node := range queue {
			queue = queue[1:]
			if node != nil {
				sum += node.Val
				if node.Left != nil {
					queue = append(queue, node.Left)
				}
				if node.Right != nil {
					queue = append(queue, node.Right)
				}
			}
		}

		if minHeap.Len() == k && sum > (*minHeap)[0] {
			heap.Pop(minHeap)
			heap.Push(minHeap, sum)
		} else if minHeap.Len() < k {
			heap.Push(minHeap, sum)
		}
	}
	if minHeap.Len() < k {
		return -1
	}
	return int64((*minHeap)[0])
}
