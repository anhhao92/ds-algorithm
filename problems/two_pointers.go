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
