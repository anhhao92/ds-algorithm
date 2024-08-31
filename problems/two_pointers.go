package problems

import (
	"math"
	"slices"
)

func SortColors(nums []int) {
	left, right, cur := 0, len(nums)-1, 0
	for cur <= right {
		if nums[cur] == 2 {
			nums[cur], nums[right] = nums[right], nums[cur]
			right--
		} else if nums[cur] == 1 {
			cur++
		} else {
			nums[cur], nums[left] = nums[left], nums[cur]
			left++
			cur++
		}
	}
}

// LC 27
func RemoveElement(nums []int, val int) int {
	l := 0
	// remove non-target with target
	// 1 2 2 4 0 target=2 ->  1 4 0 4 0
	for i := range nums {
		if nums[i] != val {
			nums[l] = nums[i]
			l++
		}
	}
	return l
}

func TwoSum(numbers []int, target int) []int {
	l, r := 0, len(numbers)
	for l < r {
		total := numbers[l] + numbers[r]
		if total == target {
			return []int{l, r}
		}
		if total > target {
			r--
		} else {
			l++
		}
	}
	return []int{-1, -1}
}

func PartitionLabels(s string) []int {
	occurrence := make(map[byte]int)
	res := []int{}
	size, end := 0, 0
	for i := 0; i < len(s); i++ {
		occurrence[s[i]] = i
	}
	for i := 0; i < len(s); i++ {
		end = max(end, occurrence[s[i]])
		size++
		if i == end {
			res = append(res, size)
			size = 0
		}
	}
	return res
}

func MaxContainerWaterArea(height []int) int {
	maxWaterArea := 0
	for l, r := 0, len(height)-1; l < r; {
		maxWaterArea = max(maxWaterArea, (r-l+1)*min(height[l], height[r]))
		if height[l] < height[r] {
			l++
		} else {
			r--
		}
	}
	return maxWaterArea
}

// LC80
func RemoveDuplicates(nums []int) int {
	count := 0
	l, r := 0, 1

	for r < len(nums) {
		if nums[l] != nums[r] {
			l++
			nums[l] = nums[r]
			count = 0
		} else if count < 1 {
			l++
			nums[l] = nums[r]
			count++
		}
		r++
	}
	return l + 1
	//1,1,2,2,3,3
	//        ^ ^
}

// LC15
func ThreeSum(nums []int) [][]int {
	n := len(nums)
	res := [][]int{}
	if n < 3 {
		return res
	}
	slices.Sort(nums)
	for i := 0; i < n && nums[i] <= 0; i++ {
		if i >= 1 && nums[i] == nums[i-1] {
			continue
		}
		low, high := i+1, n-1
		for low < high {
			sum := nums[i] + nums[low] + nums[high]
			if sum == 0 {
				res = append(res, []int{nums[i], nums[low], nums[high]})
				low++
				high--
				for low < high && nums[low] == nums[low-1] {
					low++
				}
				for low < high && nums[high] == nums[high+1] {
					high--
				}
			} else if sum < 0 {
				low++
			} else {
				high--
			}
		}
	}
	return res
}

// LC16
func ThreeSumClosest(nums []int, target int) int {
	n := len(nums)
	sum, gap := 0, math.MaxInt32
	slices.Sort(nums)

	for i := 0; i < n && nums[i] <= 0; i++ {
		low, high := i+1, n-1
		currentGap := math.MaxInt32
		for low < high {
			threeSum := nums[i] + nums[low] + nums[high]
			currentGap = int(math.Abs(float64(threeSum) - float64(target)))
			if threeSum == target {
				return threeSum
			} else if threeSum < target { // compare with targe
				low++
			} else {
				high--
			}
			if currentGap < gap {
				sum = threeSum
				gap = currentGap
			}
		}
	}
	return sum
}

// LC 18
func fourSum(nums []int, target int) [][]int {
	n := len(nums)
	res := [][]int{}
	sets := map[[4]int]bool{}
	slices.Sort(nums)
	for i := 0; i < n-3; i++ {
		for j := i + 1; j < n-2; j++ {
			newTarget := target - nums[i] - nums[j]
			low, high := j+1, n-1
			for low < high {
				if nums[low]+nums[high] > newTarget {
					high--
				} else if nums[low]+nums[high] < newTarget {
					low++
				} else {
					sets[[4]int{nums[i], nums[j], nums[low], nums[high]}] = true
					low++
					high--
				}
			}
		}
	}
	for k := range sets {
		res = append(res, k[:])
	}
	return res
}

// LC42
func TrapWater(height []int) int {
	n := len(height)
	total := 0
	maxIndex, leftMaxIndex, rightMaxIndex := 0, 0, n-1
	for i := 1; i < n; i++ {
		if height[maxIndex] < height[i] {
			maxIndex = i
		}
	}
	for i := 0; i < maxIndex; i++ {
		if height[leftMaxIndex] < height[i] {
			leftMaxIndex = i
		}
		total += min(height[leftMaxIndex], height[maxIndex]) - height[i]
	}
	for i := n - 2; i >= maxIndex; i-- {
		if height[rightMaxIndex] < height[i] {
			rightMaxIndex = i
		}
		total += min(height[rightMaxIndex], height[maxIndex]) - height[i]
	}

	return total
}

// LC1750
func minimumLengthAfterDeleting(s string) int {
	l, r := 0, len(s)-1
	for l < r && s[l] == s[r] {
		if l+1 < r && s[l] == s[l+1] {
			l++
			continue
		}
		if r-1 > l && s[r] == s[r-1] {
			r--
			continue
		}
		l++
		r--
	}
	return r - l + 1
}

// LC948
func bagOfTokensScore(tokens []int, power int) int {
	slices.Sort(tokens)
	l, r := 0, len(tokens)-1
	score, res := 0, 0
	for l <= r {
		// face-up
		if tokens[l] <= power {
			power -= tokens[l]
			score++
			l++
			res = max(res, score)
		} else if score > 0 {
			power += tokens[r]
			score--
			r--
		} else {
			break
		}
	}
	return res
}

// LC2149
func rearrangeArray(nums []int) []int {
	l, r := 0, 1
	res := make([]int, len(nums))
	for i := 0; i < len(nums); i++ {
		if nums[i] > 0 {
			res[l] = nums[i]
			l += 2
		} else {
			res[r] = nums[i]
			r += 2
		}
	}
	return res
}

// LC189
func rotate(nums []int, k int) {
	k = k % len(nums)
	slices.Reverse(nums)
	slices.Reverse(nums[:k])
	slices.Reverse(nums[k:])
}

// LC 283
func moveZeroes(nums []int) {
	l := 0
	for r := 0; r < len(nums); r++ {
		if nums[r] != 0 {
			nums[l], nums[r] = nums[r], nums[l]
			l++
		}
	}
}

// LC881
func numRescueBoats(people []int, limit int) int {
	slices.Sort(people)
	res, l, r := 0, 0, len(people)-1
	for l <= r {
		if people[r]+people[l] <= limit {
			l++
		}
		res++
		r--
	}
	return res
}

// LC779
func kthGrammar(n int, k int) int {
	current := 0
	l, r := 1, int(math.Pow(2, float64(n-1)))
	for i := 0; i < n; i++ {
		mid := (l + r) / 2
		if k <= mid {
			r = mid
		} else {
			l = mid + 1
			if current == 0 {
				current = 1
			} else {
				current = 0
			}
		}
	}
	return current
}

// LC 844
func backspaceCompare(s string, t string) bool {
	nextCharacter := func(str string, index int) int {
		count := 0
		for index >= 0 {
			if count == 0 && str[index] != '#' {
				break
			} else if str[index] == '#' {
				count++
			} else {
				count--
			}
			index--
		}
		return index
	}

	sLen, tLen := len(s)-1, len(t)-1
	for sLen >= 0 || tLen >= 0 {
		sLen = nextCharacter(s, sLen)
		tLen = nextCharacter(t, tLen)
		sChar, tChar := "", ""
		if sLen >= 0 {
			sChar = string(s[sLen])
		}
		if tLen >= 0 {
			tChar = string(t[tLen])
		}
		if sChar != tChar {
			return false
		}
		sLen--
		tLen--
	}
	return true
}

// LC 1662
func arrayStringsAreEqual(word1 []string, word2 []string) bool {
	l, r := 0, 0
	w1, w2 := 0, 0
	for w1 < len(word1) && w2 < len(word2) {
		if word1[w1][l] != word2[w2][r] {
			return false
		}
		l++
		r++
		if l == len(word1[w1]) {
			l = 0
			w1++
		}
		if r == len(word2[w2]) {
			r = 0
			w2++
		}
	}
	return w1 == len(word1) && w2 == len(word2)
}

// LC 557
func reverseWords(s string) string {
	l, chars := 0, []byte(s)
	for i := 0; i < len(chars); i++ {
		if i == len(chars)-1 || chars[i] == ' ' {
			if i == len(chars)-1 {
				i = len(chars)
			}
			for j := i - 1; j > l; j-- {
				chars[l], chars[j] = chars[j], chars[l]
				l++
			}
			l = i + 1
		}
	}
	return string(chars)
}

// LC 977
func SortedSquares(nums []int) []int {
	i := len(nums) - 1
	l, r := 0, i
	res := make([]int, len(nums))
	for l <= r {
		if abs(nums[l]) <= abs(nums[r]) {
			res[i] = nums[r] * nums[r]
			r--
		} else {
			res[i] = nums[l] * nums[l]
			l++
		}
		i--
	}
	return nums
}
