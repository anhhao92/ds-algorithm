package problems

/*
Prefix Sum + Hashmap Approach
- If the prefix sum up to i index is X, and the prefix sum up to j index is Y
  and it is found that Y = X + k, then the required subarray is found with i as start index and j as end index.
- To store the index value and the sum of elements up to that index a hashmap can be used.
What is stopping you from using sliding window?
- Sliding window is only applicable when we know for sure if the prefixsum is an increasing or decreasing
*/

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

// LC930
func numSubarraysWithSum(nums []int, goal int) int {
	psum, res := 0, 0
	count := make([]int, len(nums)+1) // this one can be use hash map
	count[0] = 1
	// The problem is finding subarray where the sum of elements is goal
	for _, num := range nums {
		psum += num
		if psum >= goal {
			res += count[psum-goal] // if we remove the elements sum up to count[psum-goal]
		}
		count[psum]++
	}
	return res
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

// LC525
func findMaxLength(nums []int) int {
	psum, res := 0, 0
	prefix := map[int]int{}
	prefix[0] = -1 // base
	// By converting each 0 to -1
	// The problem is finding the longest subarray where the sum of elements is zero
	for i := range nums {
		if nums[i] == 0 {
			psum--
		} else {
			psum++
		}
		if j, ok := prefix[psum]; ok {
			res = max(res, i-j)
		} else {
			prefix[psum] = i
		}
	}
	return res
}

// LC 523
func checkSubarraySum(nums []int, k int) bool {
	hs := map[int]int{0: -1}
	curSum := 0

	for i, val := range nums {
		curSum += val
		if j, ok := hs[curSum%k]; ok {
			// [0..i..j]
			// if we got duplicated value in the hash map it means the subarray [i,j] % k = 0
			if i-j >= 2 {
				return true
			}
		} else {
			hs[curSum%k] = i
		}
	}
	return false
}

//LC 974
func subarraysDivisibleByK(nums []int, k int) int {
	hs := map[int]int{0: 1}
	curSum, count := 0, 0
	// [-1 2 9] k = 2
	for _, val := range nums {
		curSum = (curSum + val) % k
		if curSum < 0 {
			curSum += k // ADD k if sum negative to make it positive
		}
		if v, ok := hs[curSum]; ok {
			count += v
		}
		hs[curSum]++
	}
	return count
}

// LC 1074
// 2D subarray equal target
func NumSubmatrixSumTarget(matrix [][]int, target int) int {
	row, col := len(matrix), len(matrix[0])
	// 1st colum
	for i := 1; i < row; i++ {
		matrix[i][0] += matrix[i-1][0]
	}
	// 1st row
	for j := 1; j < col; j++ {
		matrix[0][j] += matrix[0][j-1]
	}
	// the rest
	for i := 1; i < row; i++ {
		for j := 1; j < col; j++ {
			// left + top - topLeft because we add topLeft twice
			matrix[i][j] += matrix[i][j-1] + matrix[i-1][j] - matrix[i-1][j-1]
		}
	}
	count, curSum := 0, 0
	for r1 := 0; r1 < row; r1++ {
		for r2 := r1; r2 < row; r2++ {
			prefixMap := map[int]int{0: 1}
			for c := 0; c < col; c++ {
				curSum = matrix[r2][c]
				if r1 > 0 {
					curSum = matrix[r2][c] - matrix[r1-1][c]
				}
				count += prefixMap[curSum-target]
				prefixMap[curSum]++
			}
		}
	}
	return count
}

// LC 2483
func BestClosingTime(customers string) int {
	n := len(customers)
	prefix := make([]int, n+1)
	postfix := make([]int, n+1)
	penalty, index := n, 0
	for i := 1; i <= n; i++ {
		prefix[i] = prefix[i-1]
		if customers[i-1] == 'N' {
			prefix[i] += 1
		}
	}
	for i := n - 1; i >= 0; i-- {
		postfix[i] = postfix[i+1]
		if customers[i] == 'Y' {
			postfix[i] += 1
		}
	}

	for i := 0; i <= n; i++ {
		if prefix[i]+postfix[i] < penalty {
			penalty = prefix[i] + postfix[i]
			index = i
		}
	}
	return index
}
