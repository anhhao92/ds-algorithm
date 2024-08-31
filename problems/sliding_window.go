package problems

import (
	"math"
	"slices"
)

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

// LC 658
func FindClosestElements(arr []int, k int, x int) []int {
	l, r := 0, len(arr)-k
	for l < r {
		mid := l + (r-l)/2
		// [mid]-----x-[mid+k] move to the right
		// [mid]-x-----[mid+k] move to the left
		// not absolute value distance because we calculate the relative distance
		if x-arr[mid] <= arr[mid+k]-x {
			r = mid
		} else {
			l = mid + 1
		}
	}
	return arr[l : l+k]
}

// LC 219
func containsNearbyDuplicate(nums []int, k int) bool {
	prevIndex := map[int]int{}
	for i := range nums {
		if left, ok := prevIndex[nums[i]]; ok && i-left <= k {
			return true
		}
		prevIndex[nums[i]] = i
	}
	return false
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

// LC2962
func countSubarrays(nums []int, k int) int64 {
	maxK := slices.Max(nums)
	l, r, c, total := 0, 0, 0, 0
	for r < len(nums) {
		if nums[r] == maxK {
			c++
		}
		// contract window
		for l <= r && c >= k {
			if nums[l] == maxK {
				c--
			}
			total += len(nums) - r // total subarray for current window
			l++
		}
		r++
	}
	return int64(total)
}

// LC 2958
func maxSubarrayLength(nums []int, k int) int {
	count := make(map[int]int)
	l, r, res := 0, 0, 0
	for r < len(nums) {
		count[nums[r]]++
		for count[nums[r]] > k {
			count[nums[l]]--
			l++
		}
		res = max(res, r-l+1)
		r++

	}
	return res
}

// LC713
func numSubarrayProductLessThanK(nums []int, k int) int {
	l, r, res := 0, 0, 0
	total := 1
	for r < len(nums) {
		total *= nums[r]
		for l <= r && total >= k {
			total /= nums[l]
			l++
		}
		res += r - l + 1
		r++
	}
	return res
}

// LC2009
func minOperations(nums []int) int {
	slices.Sort(nums)
	length := len(nums)
	res := length
	uniqueNums := slices.Compact(nums)
	for l, r := 0, 0; l < len(uniqueNums); l++ {
		for r < len(uniqueNums) && uniqueNums[r] < uniqueNums[l]+length {
			r++
		}
		res = min(res, length-(r-l))
	}
	return res
}

// LC1658
func minOperationsReduceXToZero(nums []int, x int) int {
	target, length, curSum, l, r := 0, -1, 0, 0, 0
	for _, v := range nums {
		target += v
	}
	target = target - x
	// Find max window where sum=target
	// Because nums is array of positive numbers so we can use sliding window
	for r < len(nums) {
		curSum += nums[r]
		for l <= r && curSum > target {
			curSum -= nums[l]
			l++
		}
		if curSum == target {
			length = max(length, r-l+1)
		}
		r++
	}
	if length == -1 {
		return -1
	}
	return len(nums) - length
}

// LC992
func subarraysWithKDistinct(nums []int, k int) int {
	count := map[int]int{}
	leftFar, leftNear, res := 0, 0, 0
	for r := 0; r < len(nums); r++ {
		count[nums[r]]++
		for len(count) > k {
			count[nums[leftNear]]--
			if count[nums[leftNear]] == 0 {
				delete(count, nums[leftNear])
			}
			leftNear++
			leftFar = leftNear
		}
		for count[nums[leftNear]] > 1 {
			count[nums[leftNear]]--
			leftNear++
		}
		if len(count) == k {
			res += leftNear - leftFar + 1
		}
	}
	return res
}

// LC1456
func maxVowels(s string, k int) int {
	isVowel := map[byte]bool{'a': true, 'e': true, 'i': true, 'o': true, 'u': true}
	curLen, maxLen, l := 0, 0, 0
	for r := 0; r < len(s); r++ {
		if isVowel[s[r]] {
			curLen++
		}
		if r-l+1 > k {
			if isVowel[s[l]] {
				curLen--
			}
			l++
		}
		maxLen = max(maxLen, curLen)
	}
	return maxLen
}

// LC1838
func maxFrequencyAfterKOperations(nums []int, k int) int {
	slices.Sort(nums)
	l, r, total, maxWindow := 0, 0, 0, 0
	for r < len(nums) {
		total += nums[r]
		// make nums[r] as the most frequent number in current sliding window
		for (r-l+1)*nums[r]-total > k {
			total -= nums[l]
			l++
		}
		maxWindow = max(maxWindow, r-l+1)
		r++
	}
	return maxWindow
}

// LC1888
func minFlips(s string) int {
	n := len(s)
	str := []byte(s + s)
	atl1, atl2 := make([]byte, len(str)), make([]byte, len(str))
	for i := 0; i < len(str); i++ {
		if i%2 == 0 {
			atl2[i] = '1'
			atl1[i] = '0'
		} else {
			atl2[i] = '0'
			atl1[i] = '1'
		}
	}
	res := len(str)
	diff1, diff2, l := 0, 0, 0
	for r := 0; r < len(str); r++ {
		if str[r] != atl1[r] {
			diff1++
		}
		if str[r] != atl2[r] {
			diff2++
		}
		if r-l+1 > n {
			if str[l] != atl1[l] {
				diff1--
			}
			if str[l] != atl2[l] {
				diff2--
			}
			l++
		}
		if r-l+1 == n {
			res = min(res, diff1, diff2)
		}
	}
	return res
}

// LC 1208
func equalSubstring(s string, t string, maxCost int) int {
	diff := func(a, b byte) int {
		if a > b {
			return int(a - b)
		}
		return int(b - a)
	}
	cost, l, res := 0, 0, 0
	for r := 0; r < len(s); r++ {
		cost += diff(s[r], t[r])
		for cost > maxCost {
			cost -= diff(s[l], t[l])
			l++
		}
		res = max(res, r-l+1)
	}
	return res
}

// LC 30
func FindSubstring(s string, words []string) []int {
	dict := map[string]int{}
	wordSize := len(words[0])
	totalWord := len(words)
	res := []int{}
	for _, w := range words {
		dict[w]++
	}
	for i := 0; i < wordSize; i++ {
		left := i
		count := 0
		seen := map[string]int{}
		for j := i; j+wordSize <= len(s); j += wordSize {
			word := s[j : j+wordSize]
			if c, ok := dict[word]; ok {
				seen[word]++
				count++
				// shrink window
				for seen[word] > c {
					leftWord := s[left : left+wordSize]
					seen[leftWord]--
					count--
					left += wordSize
				}
				if count == totalWord {
					res = append(res, left)
					leftWord := s[left : left+wordSize]
					seen[leftWord]--
					count--
					left += wordSize
				}

			} else {
				clear(seen)
				count = 0
				left = j + wordSize
			}
		}
	}
	return res
}
