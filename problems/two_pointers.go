package problems

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

func twoSum(numbers []int, target int) []int {
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
