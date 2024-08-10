package problems

import "math"

// Leetcode 239
func maxSlidingWindow(nums []int, k int) []int {
	queue := []int{} // decreasing queue
	res := []int{}

	for l, r := 0, 0; r < len(nums); r++ {
		for len(queue) > 0 {
			index := queue[len(queue)-1]
			if nums[r] > nums[index] {
				queue = queue[:len(queue)-1]
			} else {
				break
			}
		}
		queue = append(queue, r)
		if l > queue[0] {
			queue = queue[1:]
		}

		if r >= k-1 {
			res = append(res, nums[queue[0]])
			l++
		}
	}
	return res
}

// s = "ADOBECODEBANC", t = "ABC"
func MinWindow(s string, t string) string {
	// define table
	chars := make([]int, 128)
	for i := 0; i < len(t); i++ {
		chars[t[i]]++
	}
	left, right, counter, minLength := 0, 0, 0, math.MaxInt32
	result := ""
	for right < len(s) {
		rightChar := s[right]
		chars[rightChar]--
		if chars[rightChar] >= 0 {
			counter++
		}
		for counter == len(t) {
			leftChar := s[left]
			size := right - left + 1
			if size < minLength {
				minLength = size
				result = s[left : right+1]
			}
			chars[leftChar]++
			if chars[leftChar] > 0 {
				counter--
			}
			left++
		}

		right++
	}
	return result
}

// s1 := "ab" s2 := "eidboaooo"
func CheckInclusion(s1 string, s2 string) bool {
	// define table
	chars := make([]int, 128)
	for i := 0; i < len(s1); i++ {
		chars[s1[i]]++
	}
	left, right, counter, minLength := 0, 0, 0, math.MaxInt32
	for right < len(s2) {
		rightChar := s2[right]
		chars[rightChar]--
		if chars[rightChar] >= 0 {
			counter++
		}
		// contract window
		for counter == len(s1) {
			size := right - left + 1
			leftChar := s2[left]
			minLength = min(minLength, size)
			chars[leftChar]++
			if chars[leftChar] > 0 {
				counter--
			}
			left++
		}
		right++

	}
	return minLength == len(s1)
}

func LongestOnes(nums []int, k int) int {
	// base case
	n := len(nums)
	if n < 2 {
		return n
	}
	// define pointer
	l, r, counter, maxLen := 0, 0, 0, 0
	for r < n {
		if nums[r] == 0 {
			counter++
		}
		// contract window if we don't meet condition
		for counter > k {
			if nums[l] == 0 {
				counter--
			}
			l++
		}
		maxLen = max(maxLen, r-l+1)
		r++
	}
	return maxLen
}

func CharacterReplacement(s string, k int) int {
	maxLen, maxFreq := 0, 0
	freq := map[byte]int{}
	for r, l := 0, 0; r < len(s); r++ {
		freq[s[r]]++
		maxFreq = max(maxFreq, freq[s[r]])
		if (r - l + 1 - maxFreq) > k {
			freq[s[l]]--
			l++
		}
		maxLen = max(maxLen, r-l+1)
	}
	return maxLen
}

// LC 209
func MinSubArrayLen(target int, nums []int) int {
	minLength, sum := len(nums), 0
	for l, r := 0, 0; r < len(nums); r++ {
		sum += nums[r]
		for sum >= target {
			minLength = min(minLength, r-l+1)
			sum -= nums[l]
			l++
		}
	}
	if minLength != math.MaxInt32 {
		return minLength
	}
	return 0
}
