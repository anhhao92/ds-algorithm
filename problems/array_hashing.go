package problems

import (
	"math"
)

func subarraySum(nums []int, k int) int {
	prefixSum := make([]int, len(nums))
	hs := map[int]int{}
	count := 0
	prefixSum[0] = nums[0]
	for i := 1; i < len(nums); i++ {
		prefixSum[i] = prefixSum[i-1] + nums[i]
	}
	for _, prefix := range prefixSum {
		if prefix == k {
			count++
		}
		if hs[prefix-k] != 0 {
			count += hs[prefix-k]
		}
		hs[prefix]++
	}
	return count
}
func LongestEvenSubsequence(nums []int) int {
	minOdd, maxOdd, minEven, maxEven := math.MaxInt32, math.MinInt32, math.MaxInt32, math.MinInt32
	for _, v := range nums {
		if v%2 == 0 {
			minEven = min(minEven, v)
			maxEven = max(maxEven, v)
		} else {
			minOdd = min(minOdd, v)
			maxOdd = max(maxOdd, v)
		}
	}

	odd, even := 0, 0
	for _, v := range nums {
		if minEven <= v && v <= maxEven {
			even++
		}
		if minOdd <= v && v <= maxOdd {
			odd++
		}
	}

	return max(odd, even)
}

func FarestDistanceBetweenZeroOne(nums []int) int {
	j, count := 0, 0
	for i := 1; i < len(nums); i++ {
		if nums[i] == 1 {
			if nums[j] == 1 {
				count = max(count, i-j)
			}
			j = i
		}
	}
	return count
}

func TopKFrequent(nums []int, k int) []int {
	hs := make(map[int]int)
	res := []int{}
	for _, v := range nums {
		hs[v]++
	}
	// bucket sort
	bucket := make(map[int][]int)
	maxFreq := 1
	for key, value := range hs {
		bucket[value] = append(bucket[value], key)
		maxFreq = max(maxFreq, value)
	}
	for i := maxFreq; i > 0 && k > 0; i-- {
		if value, hasKey := bucket[i]; hasKey {
			res = append(res, value...)
			k -= len(value)
		}
	}
	return res
}

func FrequencySort(s string) string {
	res := []byte{}
	hs := make(map[byte]int)
	for i := 0; i < len(s); i++ {
		hs[s[i]]++
	}
	// bucket sort
	bucket := make(map[int][]byte)
	for key, value := range hs {
		bucket[value] = append(bucket[value], key)
	}
	for i := len(s); i > 0; i-- {
		if value, ok := bucket[i]; ok {
			for _, c := range value {
				for j := 0; j < i; j++ {
					res = append(res, c)
				}
			}
		}
	}
	return string(res)
}
