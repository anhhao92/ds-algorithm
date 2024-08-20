package problems

import (
	"container/heap"
)

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
