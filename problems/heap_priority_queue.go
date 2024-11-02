package problems

import (
	"container/heap"
	"math"
	"slices"
	"strings"
)

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

// 973
func kClosest(points [][]int, k int) [][]int {
	type Point struct {
		dist  int
		index int
	}
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

// LC 215
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

// LC 1985
func kthLargestNumber(nums []string, k int) string {
	comparator := func(a, b string) bool {
		if len(a) != len(b) {
			return len(a) > len(b)
		}
		return a > b
	}
	maxHeap := Heap[string]{values: nums, comparator: comparator}
	heap.Init(&maxHeap)
	for k > 1 {
		heap.Pop(&maxHeap)
		k--
	}
	return maxHeap.Peak()
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

// LC 1046
func lastStoneWeight(stones []int) int {
	maxHeap := &Heap[int]{comparator: func(i, j int) bool { return i > j }, values: stones}
	heap.Init(maxHeap)
	for maxHeap.Len() > 1 {
		v1 := heap.Pop(maxHeap).(int)
		v2 := heap.Pop(maxHeap).(int)
		heap.Push(maxHeap, abs(v1-v2))
	}
	return heap.Pop(maxHeap).(int)
}

// LC 1481
func findLeastNumOfUniqueInts(arr []int, k int) int {
	freq := map[int]int{}
	for _, v := range arr {
		freq[v]++
	}
	minHeap := IntHeap{}
	heap.Init(&minHeap)
	for _, v := range freq {
		heap.Push(&minHeap, v)
	}

	for minHeap.Len() > 0 && k > 0 {
		frq := minHeap[0]
		if frq <= k {
			heap.Pop(&minHeap)
		}
		k -= frq
	}
	return minHeap.Len()
}

// LC 1642
func furthestBuilding(heights []int, bricks int, ladders int) int {
	maxHeap := IntHeap{}
	heap.Init(&maxHeap)
	for i := 0; i < len(heights)-1; i++ {
		diff := heights[i+1] - heights[i]
		if diff <= 0 {
			continue
		}
		bricks -= diff
		heap.Push(&maxHeap, -diff)
		if bricks < 0 {
			if ladders == 0 {
				return i
			}
			ladders--
			bricks += -heap.Pop(&maxHeap).(int)
		}
	}
	return len(heights) - 1
}

// LC 767
func reorganizeString(s string) string {
	var freq [26]int
	for i := 0; i < len(s); i++ {
		freq[s[i]-'a']++
	}
	comparator := func(i, j int) bool {
		return freq[i] > freq[j]
	}
	maxHeap := &Heap[int]{comparator: comparator}
	heap.Init(maxHeap)
	for i, count := range freq {
		if count != 0 {
			heap.Push(maxHeap, i)
		}
	}
	var prev int = -1
	var res strings.Builder
	// a:3 b:2 c:2 try to be greedy max freq
	for maxHeap.Len() > 0 {
		i := heap.Pop(maxHeap).(int)
		freq[i]--
		res.WriteByte(byte(i + 'a'))
		// push prev to max heap to use it in the next iteration
		if prev != -1 {
			heap.Push(maxHeap, prev)
			prev = -1
		}
		if freq[i] != 0 {
			prev = i
		}
	}

	if prev != -1 {
		return ""
	}
	return res.String()
}

// LC 1405
func longestDiverseString(a int, b int, c int) string {
	freq := []int{a, b, c}
	comparator := func(i, j int) bool {
		return freq[i] > freq[j]
	}
	maxHeap := &Heap[int]{comparator: comparator}
	heap.Init(maxHeap)
	for i, v := range freq {
		if v != 0 {
			heap.Push(maxHeap, i)
		}
	}
	var sb strings.Builder
	var prev1, prev2 byte
	for maxHeap.Len() > 0 {
		i := heap.Pop(maxHeap).(int)
		c := byte(i + 'a')
		if c == prev1 && c == prev2 {
			if maxHeap.Len() == 0 {
				break
			}
			// pop another character
			j := heap.Pop(maxHeap).(int)
			c2 := byte(j + 'a')
			freq[j]--
			sb.WriteByte(c2)
			prev1 = prev2
			prev2 = c2
			if freq[j] != 0 {
				heap.Push(maxHeap, j)
			}
		} else {
			freq[i]--
			sb.WriteByte(c)
			prev1 = prev2
			prev2 = c
		}
		if freq[i] != 0 {
			heap.Push(maxHeap, i)
		}
	}
	return sb.String()
}

// LC 1834 Single-Threaded CPU
func getOrder(tasks [][]int) []int {
	taskIndices := make([][3]int, len(tasks))
	for i, v := range tasks {
		taskIndices[i] = [3]int{v[0], v[1], i} // enqueue time, processing time, index
	}
	// sort by enqueue time
	slices.SortFunc(taskIndices, func(a, b [3]int) int {
		return a[0] - b[0]
	})
	// minHeap by processing time
	comparator := func(a, b [3]int) bool {
		// same processing time then choose the task with the smallest index
		if a[1] == b[1] {
			return a[2] < b[2]
		}
		return a[1] < b[1]
	}
	minHeap := &Heap[[3]int]{comparator: comparator}
	heap.Init(minHeap)
	res := make([]int, 0, len(tasks))
	time := taskIndices[0][0]
	i := 0
	for minHeap.Len() > 0 || i < len(taskIndices) {
		for i < len(taskIndices) && taskIndices[i][0] <= time {
			heap.Push(minHeap, taskIndices[i])
			i++
		}
		if minHeap.Len() > 0 {
			t := heap.Pop(minHeap).([3]int)
			processingTime, index := t[1], t[2]
			time += processingTime
			res = append(res, index)
		} else {
			// CPU idle move to the next processing time
			time = taskIndices[i][0]
		}
	}
	return res
}

// LC 1882
func assignTasks(servers []int, tasks []int) []int {
	serverIndices := make([]int, len(servers))
	for i := range servers {
		serverIndices[i] = i
	}
	availableServer := &Heap[int]{comparator: func(i, j int) bool {
		if servers[i] == servers[j] {
			return i < j
		}
		return servers[i] < servers[j]
	}, values: serverIndices}
	unavailableServer := &Heap[[2]int]{comparator: func(a, b [2]int) bool {
		return a[0] < b[0]
	}} // [time, index]
	heap.Init(availableServer)
	heap.Init(unavailableServer)

	res := make([]int, len(tasks))
	time := 0
	for i := range tasks {
		time = max(time, i)
		if availableServer.Len() == 0 {
			time = unavailableServer.Peak()[0]
		}
		for unavailableServer.Len() > 0 && time >= unavailableServer.Peak()[0] {
			s := heap.Pop(unavailableServer).([2]int)
			serverIdx := s[1]
			heap.Push(availableServer, serverIdx)
		}
		serverIdx := heap.Pop(availableServer).(int)
		res[i] = serverIdx
		heap.Push(unavailableServer, [2]int{time + tasks[i], serverIdx})
	}

	return res
}

// LC 1094
func carPooling(trips [][]int, capacity int) bool {
	slices.SortFunc(trips, func(a, b []int) int {
		return a[1] - b[1]
	})
	minHeap := &Heap[[]int]{comparator: func(a, b []int) bool {
		return a[2] < b[2]
	}}
	heap.Init(minHeap)
	for _, t := range trips {
		passenger, from := t[0], t[1]
		for minHeap.Len() > 0 && minHeap.Peak()[2] <= from {
			drop := heap.Pop(minHeap).([]int)
			capacity += drop[0]
		}
		if passenger > capacity {
			return false
		}
		heap.Push(minHeap, t)
		capacity -= passenger
	}
	return true
}

// LC 1383
func maxPerformance(n int, speed []int, efficiency []int, k int) int {
	efficiencySpeed := make([][2]int, len(speed))
	for i := range n {
		efficiencySpeed[i] = [2]int{efficiency[i], speed[i]}
	}
	slices.SortFunc(efficiencySpeed, func(a, b [2]int) int {
		return b[0] - a[0]
	})
	minHeap := &Heap[int]{comparator: func(a, b int) bool { return a < b }}
	heap.Init(minHeap)
	res, speedSoFar := 0, 0
	for _, v := range efficiencySpeed {
		if minHeap.Len() == k {
			speedSoFar -= heap.Pop(minHeap).(int)
		}
		curEfficiency, curSpeed := v[0], v[1]
		speedSoFar += curSpeed
		heap.Push(minHeap, curSpeed)
		res = max(res, curEfficiency*speedSoFar)
	}
	return res % MOD
}

// LC 502
func findMaximizedCapital(k int, w int, profits []int, capital []int) int {
	capProfit := make([][2]int, len(profits))
	for i, p := range profits {
		capProfit[i] = [2]int{capital[i], p}
	}
	slices.SortFunc(capProfit, func(a, b [2]int) int {
		return a[0] - b[0]
	})
	maxHeap := &Heap[int]{comparator: func(a, b int) bool { return a > b }}
	heap.Init(maxHeap)
	var i int
	for range k {
		for i < len(capProfit) && capProfit[i][0] <= w {
			heap.Push(maxHeap, capProfit[i][1])
			i++
		}
		if maxHeap.Len() == 0 {
			break
		}
		w += heap.Pop(maxHeap).(int)
	}
	return w
}

// LC 2251
func fullBloomFlowers(flowers [][]int, people []int) []int {
	res := make([]int, len(people))
	peopleIndices := make([][2]int, len(people))
	for i, p := range people {
		peopleIndices[i] = [2]int{p, i}
	}
	slices.SortFunc(flowers, func(a, b []int) int {
		return a[0] - b[0]
	})
	slices.SortFunc(peopleIndices, func(a, b [2]int) int {
		return a[0] - b[0]
	})
	minHeap := &Heap[int]{comparator: func(a, b int) bool { return a < b }}
	heap.Init(minHeap)
	var j int
	for _, p := range peopleIndices {
		index, peoplePos := p[1], p[0]
		for ; j < len(flowers) && flowers[j][0] <= peoplePos; j++ {
			heap.Push(minHeap, flowers[j][1])
		}
		for minHeap.Len() > 0 && minHeap.Peak() < peoplePos {
			heap.Pop(minHeap)
		}
		res[index] = minHeap.Len()
	}
	return res
}

// LC 1425 DP + maxHeap
func constrainedSubsetSum(nums []int, k int) int {
	res := nums[0]
	maxHeap := &Heap[[2]int]{comparator: func(a, b [2]int) bool { return a[0] > b[0] }, values: [][2]int{{nums[0], 0}}}
	heap.Init(maxHeap)
	for i := 1; i < len(nums); i++ {
		for i-maxHeap.Peak()[1] > k {
			heap.Pop(maxHeap)
		}
		current := max(nums[i], nums[i]+maxHeap.Peak()[0])
		heap.Push(maxHeap, [2]int{current, i})
		res = max(res, current)
	}
	return res
}

// LC 2542
func maxScoreSubsequence(nums1 []int, nums2 []int, k int) int64 {
	arr := make([][2]int, len(nums1))
	for i, n1 := range nums1 {
		arr[i] = [2]int{n1, nums2[i]}
	}
	slices.SortFunc(arr, func(a, b [2]int) int {
		return b[1] - a[1]
	})
	minHeap := &Heap[int]{comparator: func(a, b int) bool { return a < b }}
	heap.Init(minHeap)
	res, curSum := 0, 0
	for _, v := range arr {
		n1, n2 := v[0], v[1]
		curSum += n1
		heap.Push(minHeap, n1)
		if minHeap.Len() > k {
			curSum -= heap.Pop(minHeap).(int)
		}
		if minHeap.Len() == k {
			res = max(res, n2*curSum)
		}
	}
	return int64(res)
}

// LC 857
func mincostToHireWorkers(quality []int, wage []int, k int) float64 {
	pairs := make([][2]float64, len(quality))
	for i, q := range quality {
		pairs[i] = [2]float64{float64(wage[i]) / float64(q), float64(q)}
	}
	slices.SortFunc(pairs, func(a, b [2]float64) int {
		if a[0] < b[0] {
			return -1
		}
		return 1
	})
	maxHeap := &Heap[float64]{comparator: func(a, b float64) bool { return a > b }}
	heap.Init(maxHeap)
	totalQuality, res := 0.0, math.MaxFloat32
	for _, v := range pairs {
		ratio, q := v[0], v[1]
		totalQuality += q
		heap.Push(maxHeap, q)
		if maxHeap.Len() > k {
			totalQuality -= heap.Pop(maxHeap).(float64)
		}
		if maxHeap.Len() == k {
			res = min(res, totalQuality*ratio)
		}

	}
	return res
}

// LC 1675
func minimumDeviation(nums []int) int {
	arr := make([][2]int, len(nums))
	maxVal := 0
	for i, n := range nums {
		originVal := n
		for n%2 == 0 {
			n /= 2
		}
		arr[i] = [2]int{n, max(originVal, 2*n)}
		maxVal = max(maxVal, n)
	}
	minHeap := &Heap[[2]int]{comparator: func(a, b [2]int) bool { return a[0] < b[0] }}
	heap.Init(minHeap)
	res := math.MaxInt32
	for minHeap.Len() == len(nums) {
		v := heap.Pop(minHeap).([2]int)
		num, maxNum := v[0], v[1]
		res = min(res, maxVal-num)
		if num < maxNum {
			heap.Push(minHeap, [2]int{num * 2, maxNum})
			maxVal = max(maxVal, num*2)
		}
	}
	return res
}

// LC 355
type (
	Tweet struct {
		timestamp int
		tweetId   int
	}
	Twitter struct {
		timestamp int
		tweetMap  map[int][]Tweet
		followMap map[int]map[int]bool
	}
)

func TwitterConstructor() Twitter {
	return Twitter{tweetMap: make(map[int][]Tweet), followMap: make(map[int]map[int]bool)}
}

func (this *Twitter) PostTweet(userId int, tweetId int) {
	this.tweetMap[userId] = append(this.tweetMap[userId], Tweet{this.timestamp, tweetId})
	this.timestamp++
}

func (this *Twitter) GetNewsFeed(userId int) []int {
	maxHeap := Heap[[4]int]{comparator: func(a, b [4]int) bool {
		return a[0] > b[0]
	}}
	heap.Init(&maxHeap)
	followedUsers := this.followMap[userId]
	if followedUsers == nil {
		followedUsers = make(map[int]bool)
	}
	followedUsers[userId] = true // user follows themself

	for followee := range followedUsers {
		if tweets, ok := this.tweetMap[followee]; ok {
			n := len(tweets) - 1
			t := tweets[n]
			heap.Push(&maxHeap, [4]int{t.timestamp, t.tweetId, followee, n - 1})
		}
	}
	newsFeed := make([]int, 0, 10)
	for maxHeap.Len() > 0 && len(newsFeed) < 10 {
		t := heap.Pop(&maxHeap).([4]int)
		tweetId, followerId, index := t[1], t[2], t[3]
		newsFeed = append(newsFeed, tweetId)
		if index >= 0 {
			tweets := this.tweetMap[followerId]
			nextTweet := tweets[index]
			heap.Push(&maxHeap, [4]int{nextTweet.timestamp, nextTweet.tweetId, followerId, index - 1})
		}
	}
	return newsFeed
}

func (this *Twitter) Follow(followerId int, followeeId int) {
	if this.followMap[followerId] == nil {
		this.followMap[followerId] = make(map[int]bool)
	}
	this.followMap[followerId][followeeId] = true

}

func (this *Twitter) Unfollow(followerId int, followeeId int) {
	if this.followMap[followerId] != nil {
		delete(this.followMap[followerId], followeeId)
	}
}
