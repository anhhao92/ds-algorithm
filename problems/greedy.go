package problems

import (
	"container/heap"
	"math"
	"slices"
	"strings"
)

// LC 2846
func maximumOddBinaryNumber(s string) string {
	count, n := 0, len(s)
	for i := 0; i < n; i++ {
		if s[i] == '1' {
			count++
		}
	}
	res := make([]byte, n)
	for i := range res {
		if i < count-1 {
			res[i] = '1'
		} else {
			res[i] = '0'
		}
	}
	res[n-1] = '1'
	return string(res)
}

func maxSubarraySumCircular(nums []int) int {
	globalMin, globalMax := nums[0], nums[0]
	curMin, curMax, sum := 0, 0, 0
	for _, v := range nums {
		curMax = max(curMax+v, v)
		curMin = min(curMin+v, v)
		sum += v
		globalMax = max(globalMax, curMax)
		globalMin = min(globalMin, curMin)

	}
	if globalMax > 0 {
		return max(globalMax, sum-globalMin)
	}
	return globalMax
}

func maxTurbulenceSize(arr []int) int {
	left, right := 0, 1
	maxLen := 1
	prev := '*'
	for right < len(arr) {
		if arr[right-1] > arr[right] && prev != '>' {
			maxLen = max(maxLen, right-left+1)
			right++
			prev = '>'
		} else if arr[right-1] < arr[right] && prev != '<' {
			maxLen = max(maxLen, right-left+1)
			right++
			prev = '<'
		} else {
			if arr[right-1] == arr[right] {
				right++
			}
			prev = '*'
			left = right - 1
		}
	}
	return maxLen
}
func canCompleteCircuit(gas []int, cost []int) int {
	sumGas, sumCost := 0, 0
	for i := 0; i < len(gas); i++ {
		sumCost += cost[i]
		sumGas += gas[i]
	}
	if sumGas < sumCost {
		return -1
	}
	start, total := 0, 0
	for i := 0; i < len(gas); i++ {
		total += gas[i] - cost[i]
		if total < 0 {
			total = 0
			start = i + 1
		}
	}
	return start
}

func IsNStraightHand(hand []int, groupSize int) bool {
	if len(hand)%groupSize != 0 {
		return false
	}
	occurrence := make(map[int]int)
	for i := 0; i < len(hand); i++ {
		occurrence[hand[i]]++
	}
	minHeap := []int{}
	for k := range occurrence {
		minHeap = append(minHeap, k)
	}
	h := IntHeap(minHeap)
	heap.Init(&h)
	for h.Len() > 0 {
		minValue := h[0]
		for j := minValue; j < minValue+groupSize; j++ {
			if _, hasKey := occurrence[j]; !hasKey {
				return false
			}
			occurrence[j]--
			if occurrence[j] == 0 {
				heap.Pop(&h)
				delete(occurrence, j)
			}
		}
	}
	return true
}

// LC55
func JumpGame(nums []int) bool {
	goal := nums[len(nums)-1]
	for i := len(nums) - 1; i >= 0; i-- {
		if i+nums[i] >= goal {
			goal = i
		}

	}
	return goal == 0
}

// LC 42
func JumpGameII(nums []int) int {
	c, l, h := 0, 0, 0
	for h < len(nums)-1 {
		farthest := 0
		for i := l; i < h+1; i++ {
			farthest = max(farthest, i+nums[i])
		}
		l = h + 1
		h = farthest
		c++
	}
	return c
}

// LC 1306 BFS
func JumpGameIII(arr []int, start int) bool {
	queue := []int{start}
	visited := map[int]bool{}
	for len(queue) > 0 {
		i := queue[0]
		queue = queue[1:]
		if arr[i] == 0 {
			return true
		}
		visited[i] = true
		if i+arr[i] < len(arr) && !visited[i+arr[i]] {
			queue = append(queue, i+arr[i])
		}
		if i-arr[i] >= 0 && !visited[i-arr[i]] {
			queue = append(queue, i-arr[i])
		}
	}
	return false
}

// LC 1345 BFS
func JumpGameIV(arr []int) int {
	hash := map[int][]int{}
	for i, v := range arr {
		hash[v] = append(hash[v], i)
	}
	queue := []int{0}
	totalJump := 0
	n := len(arr)
	for len(queue) > 0 {
		size := len(queue)
		for _, index := range queue {
			if index == n-1 {
				return totalJump
			}
			if neighbors, ok := hash[arr[index]]; ok {
				for _, idx := range neighbors {
					if idx != index {
						queue = append(queue, idx)
					}
				}
				delete(hash, arr[index])
			}
			// if index+1 exist in hash table we still visit it yet
			if index+1 < n && len(hash[arr[index+1]]) > 0 {
				queue = append(queue, index+1)
			}
			if index-1 >= 0 && len(hash[arr[index-1]]) > 0 {
				queue = append(queue, index-1)
			}
		}
		queue = queue[size:]
		totalJump++
	}
	return totalJump
}

// LC 1696
func JumpGameVI(nums []int, k int) int {
	n := len(nums)
	queue := []int{0} // decreasing queue
	for i := 1; i < n; i++ {
		nums[i] += nums[queue[0]]
		for len(queue) > 0 && nums[i] > nums[queue[len(queue)-1]] {
			queue = queue[:len(queue)-1]
		}
		queue = append(queue, i)
		// window out of bound
		if i-queue[0] >= k {
			queue = queue[1:]
		}
	}
	return nums[n-1]
}

// LC 1871
func JumpGameVII(s string, minJump int, maxJump int) bool {
	queue := []int{0}
	farthest := 0
	for len(queue) > 0 {
		i := queue[0]
		queue = queue[1:]
		// to avoid duplicated jump becasue we should start farthest+1 because we reached farthest indice
		start := max(i+minJump, farthest+1)
		for j := start; j < min(len(s), i+maxJump+1); j++ {
			if s[j] == '0' {
				if j == len(s)-1 {
					return true
				}
				queue = append(queue, j)
			}
		}
		farthest = i + maxJump
	}
	return false
}

// LC 1340
func JumpGameV(arr []int, d int) int {
	n := len(arr)
	dp := make([]int, n)
	sortedIndices := make([]int, n)
	for i := range arr {
		sortedIndices[i] = i
	}
	slices.SortFunc(sortedIndices, func(i, j int) int {
		return arr[i] - arr[j]
	})
	// we have to sort the arr and go from smallest index because for array [7 6 5 4 3 2 1]
	// if arr[i] > arr[j] so dp[i] can be derived from (1+d[j])
	for _, i := range sortedIndices {
		dp[i] = 1
		// go forward
		for j := i + 1; j <= min(i+d, n-1); j++ {
			if arr[i] <= arr[j] {
				break
			}
			dp[i] = max(dp[i], 1+dp[j])
		}
		// go prev
		for j := i - 1; j >= max(i-d, 0); j-- {
			if arr[i] <= arr[j] {
				break
			}
			dp[i] = max(dp[i], 1+dp[j])
		}
	}
	return slices.Max(dp)
}

// LC 1423
func MaxScore(cardPoints []int, k int) int {
	result := 0
	l, r := 0, len(cardPoints)-k
	for i := r; i < len(cardPoints); i++ {
		result += cardPoints[i]
	}
	curSum := result
	for r < len(cardPoints) {
		curSum += cardPoints[l] - cardPoints[r]
		result = max(result, curSum)
		l++
		r++
	}
	return result
}

// LC 678
func CheckValidString(s string) bool {
	minOpen, maxOpen := 0, 0
	for i := 0; i < len(s); i++ {
		if s[i] == '(' {
			maxOpen++
			minOpen++
		} else if s[i] == ')' {
			maxOpen--
			minOpen--
		} else {
			minOpen-- // treat as ')'
			maxOpen++ // treat as '('
		}
		if maxOpen < 0 { // more '(' than ')' or '*'
			return false
		}
		if minOpen < 0 {
			minOpen = 0
		}
	}
	return minOpen == 0
}

// LC853
func carFleet(target int, position []int, speed []int) int {
	time := make([]float32, target+1)
	for i := range position {
		time[position[i]] = float32(target-position[i]) / float32(speed[i])
	}
	maxCurrent, count := float32(0), 0
	for i := len(time) - 1; i >= 0; i-- {
		if time[i] > maxCurrent {
			maxCurrent = time[i]
			count++
		}
	}
	return count
}

// LC 1921
func eliminateMaximum(dist []int, speed []int) int {
	time := make([]int, len(dist))
	for i, d := range dist {
		time[i] = int(math.Ceil(float64(d) / float64(speed[i])))
	}
	slices.Sort(time)
	for i, minute := range time {
		if i >= minute {
			return i
		}
	}
	return len(dist)
}

// LC 2439
func minimizeArrayValue(nums []int) int {
	res, total := nums[0], nums[0]
	for i := 1; i < len(nums); i++ {
		total += nums[i]
		avg := math.Ceil(float64(total) / float64(i+1))
		res = max(res, int(avg))
	}
	return res
}

// LC 1029
func twoCitySchedCost(costs [][]int) int {
	diff := make([]int, len(costs))
	res := 0
	for i, v := range costs {
		diff[i] = v[1] - v[0]
		res += v[0] // sent all people to A
	}
	slices.Sort(diff)
	// diff[i] < 0 refund or diff[i] > 0 pay more if we send person to B
	// if we want to minimize cost we have to sort diff so the refund values came first
	for i := 0; i < len(costs)/2; i++ {
		res += diff[i]
	}
	return res
}

// LC 646
func findLongestChain(pairs [][]int) int {
	// sort by ending
	slices.SortFunc(pairs, func(a, b []int) int {
		return a[1] - b[1]
	})
	count := 1
	end := pairs[0][1]
	for i := 1; i < len(pairs); i++ {
		// if 2 pairs don't overlap each other
		if end < pairs[i][0] {
			end = pairs[i][1]
			count++
		}
	}
	return count
}

// LC 1647
func minDeletionsToMakeFreqUnique(s string) int {
	hashMap := map[byte]int{}
	usedFreq := map[int]bool{}
	res := 0
	for i := 0; i < len(s); i++ {
		hashMap[s[i]]++
	}

	for _, freq := range hashMap {
		for freq > 0 && usedFreq[freq] {
			freq--
			res++
		}
		usedFreq[freq] = true
	}
	return res
}

// LC 135
func candy(ratings []int) int {
	candies := make([]int, len(ratings))
	for i := range candies {
		candies[i] = 1
	}
	for i := 1; i < len(ratings); i++ {
		if ratings[i] > ratings[i-1] {
			candies[i] = candies[i-1] + 1
		}
	}
	for i := len(ratings) - 2; i >= 0; i-- {
		if ratings[i] > ratings[i+1] && candies[i] <= candies[i+1] {
			candies[i] = candies[i+1] + 1
		}
	}
	sum := 0
	for _, v := range candies {
		sum += v
	}
	return sum
}

// LC 1846
func maximumElementAfterDecrementingAndRearranging(arr []int) int {
	slices.Sort(arr)
	prev := 0
	// 0 [2 5 7] -> [1 2 3]
	for i := 0; i < len(arr); i++ {
		prev = min(prev+1, arr[i])
	}
	return prev
}

// LC 2125
func numberOfBeams(bank []string) int {
	prev, res := 0, 0
	for i := 0; i < len(bank); i++ {
		c := strings.Count(bank[i], "1")
		if c > 0 {
			res += prev * c
			prev = c
		}
	}
	return res
}

// LC 950
func deckRevealedIncreasing(deck []int) []int {
	slices.Sort(deck)
	res := make([]int, len(deck))
	queue := make([]int, len(deck))
	for i := range queue {
		queue[i] = i
	}
	for _, v := range deck {
		i := queue[0]
		queue = queue[1:]
		res[i] = v
		if len(queue) > 0 {
			// move next value to the end of the queue
			next := queue[0]
			copy(queue, queue[1:])
			queue[len(queue)-1] = next
		}
	}
	return res
}

// LC 861
func matrixScore(grid [][]int) int {
	row, col := len(grid), len(grid[0])
	// Flip row
	for i := 0; i < row; i++ {
		if grid[i][0] == 0 {
			for j := 0; j < col; j++ {
				if grid[i][j] == 0 {
					grid[i][j] = 1
				} else {
					grid[i][j] = 0
				}
			}
		}
	}
	// Flip column
	for c := 0; c < col; c++ {
		one := 0
		for r := 0; r < row; r++ {
			one += grid[r][c]
		}
		if one < row-one {
			for i := 0; i < row; i++ {
				if grid[i][c] == 0 {
					grid[i][c] = 1
				} else {
					grid[i][c] = 0
				}
			}
		}
	}
	res := 0
	for i := 0; i < row; i++ {
		for j := 0; j < col; j++ {
			res += grid[i][j] << (col - j - 1)
		}
	}
	return res
}

// LC 1793
func maximumScoreGoodArray(nums []int, k int) int {
	l, r := k, k
	res, minValue := nums[k], nums[k]
	for l > 0 || r < len(nums)-1 {
		leftValue, rightValue := 0, 0
		if l > 0 {
			leftValue = nums[l-1]
		}
		if r < len(nums)-1 {
			rightValue = nums[r+1]
		}
		// choose right
		if leftValue < rightValue {
			r++
			minValue = min(minValue, nums[r])
		} else {
			l--
			minValue = min(minValue, nums[l])
		}
		res = max(res, (r-l+1)*minValue)
	}
	return res
}

// LC 3068
func maximumValueSum(nums []int, k int, edges [][]int) int64 {
	maxSum, count, minDiff := 0, 0, math.MaxInt32
	for _, v := range nums {
		num := v ^ k
		maxSum += max(v, num)
		minDiff = min(minDiff, abs(v-num))
		if num > v {
			count++
		}
	}
	return int64(maxSum - (count%2)*minDiff)
}

// LC 649
func predictPartyVictory(senate string) string {
	dQueue, rQueue := []int{}, []int{}
	for i := 0; i < len(senate); i++ {
		if senate[i] == 'D' {
			dQueue = append(dQueue, i)
		} else {
			rQueue = append(rQueue, i)
		}
	}
	for len(rQueue) > 0 && len(dQueue) > 0 {
		dIdx := dQueue[0]
		rIdx := rQueue[0]
		dQueue = dQueue[1:]
		rQueue = rQueue[1:]
		if rIdx < dIdx {
			rQueue = append(rQueue, dIdx+len(senate))
		} else {
			dQueue = append(dQueue, rIdx+len(senate))
		}

	}
	if len(rQueue) > 0 {
		return "Radiant"
	}
	return "Dire"
}

// LC 2038
func winnerOfGame(colors string) bool {
	alice, bob := 0, 0
	for i := 1; i < len(colors)-1; i++ {
		if colors[i-1] == colors[i] && colors[i] == colors[i+1] {
			if colors[i] == 'A' {
				alice++
			} else {
				bob++
			}
		}
	}
	return alice > bob
}

// LC 1899
func mergeTriplets(triplets [][]int, target []int) bool {
	hashSet := map[int]bool{}
	for _, t := range triplets {
		if t[0] > target[0] || t[1] > target[1] || t[2] > target[2] {
			continue
		}
		for i, v := range t {
			if v == target[i] {
				hashSet[i] = true
			}
		}
	}
	return len(hashSet) == 3
}

// LC 1968
func rearrangeArrayNotEqualAverageNeigbors(nums []int) []int {
	slices.Sort(nums)
	res := make([]int, 0, len(nums))
	l, r := 0, len(nums)-1
	for l <= r {
		res = append(res, nums[l])
		l++
		if l <= r {
			res = append(res, nums[r])
			r--
		}
	}
	return res
}
