package problems

import (
	"math"
)

// LC303
type NumArray struct {
	prefixSum []int
}

/**
 * Your NumArray object will be instantiated and called as such:
 * obj := Constructor(nums);
 * param_1 := obj.SumRange(left,right);
 */

func NewNumArray(nums []int) NumArray {
	prefixSum := make([]int, len(nums))
	prefixSum[0] = nums[0]
	for i := 1; i < len(nums); i++ {
		prefixSum[i] = prefixSum[i-1] + nums[i]
	}
	return NumArray{prefixSum: prefixSum}
}

func (this *NumArray) SumRange(left int, right int) int {
	if left > 0 {
		return this.prefixSum[right] - this.prefixSum[left-1]
	}
	return this.prefixSum[right]
}

// LC304
type NumMatrix struct {
	prefixSum [][]int
}

func NewNumMatrix(matrix [][]int) NumMatrix {
	row, col := len(matrix), len(matrix[0])
	prefixSum := make([][]int, row+1)
	for i := range prefixSum {
		prefixSum[i] = make([]int, col+1)
	}
	for i := 0; i < row; i++ {
		prefix := 0
		for j := 0; j < col; j++ {
			prefix += matrix[i][j]
			prefixSum[i+1][j+1] = prefix + prefixSum[i][j+1]
		}
	}
	return NumMatrix{prefixSum: prefixSum}
}

func (this *NumMatrix) SumRegion(row1 int, col1 int, row2 int, col2 int) int {
	bottomRight := this.prefixSum[row2+1][col2+1]
	above := this.prefixSum[row1][col2+1]
	left := this.prefixSum[row2+1][col1]
	topLeft := this.prefixSum[row1][col1]
	return bottomRight - above - left + topLeft
}

// LC560
func SubarraySum(nums []int, k int) int {
	curSum, count := 0, 0
	hs := map[int]int{0: 1}
	for _, val := range nums {
		curSum += val
		if hs[curSum-k] > 0 {
			count += hs[curSum-k]
		}
		hs[curSum]++
	}
	return count
}

// LC724
func findPivotIndex(nums []int) int {
	leftSum, total := 0, 0
	for _, v := range nums {
		total += v
	}
	for i, num := range nums {
		rightSum := total - num - leftSum
		if rightSum == leftSum {
			return i
		}
		leftSum += num
	}
	return -1
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

// LC238
func ProductExceptSelf(nums []int) []int {
	n := len(nums)
	result := make([]int, n)
	result[0] = 1
	for i := 1; i < n; i++ {
		result[i] = result[i-1] * nums[i-1]
	}
	postfix := 1
	for i := n - 1; i >= 0; i-- {
		result[i] *= postfix
		postfix *= nums[i]
	}
	return result
}

// LC128
func longestConsecutive(nums []int) int {
	setInt := make(map[int]bool)
	maxLen := 0
	for _, v := range nums {
		setInt[v] = true
	}
	for _, v := range nums {
		if !setInt[v-1] {
			length := 0
			for setInt[v+length] {
				length++
			}
			maxLen = max(maxLen, length)
		}
	}
	return maxLen
}
