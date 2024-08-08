package problems

// find value in sorted array
func SearchSortedArray(nums []int, target int) int {
	left, right := 0, len(nums)-1

	for left <= right {
		mid := left + (right-left)/2
		if nums[mid] == target {
			return mid
		}
		if nums[left] <= nums[mid] {
			if target >= nums[left] && target < nums[mid] {
				right = mid - 1
			} else {
				left = mid + 1
			}
		} else {
			if nums[mid] > target && target <= nums[right] {
				left = mid + 1
			} else {
				right = mid - 1
			}
		}
	}
	return -1
}

// 153. Find min sorted array
func FindMin(nums []int) int {
	left, right := 0, len(nums)-1
	for left+1 < right {
		mid := left + (right-left)/2
		if nums[mid] > nums[right] {
			left++
		} else {
			right--
		}
	}
	return min(nums[left], nums[right])
}

// 744. Find Smallest Letter Greater Than Target
func NextGreatestLetter(letters []byte, target byte) byte {
	left, right := 0, len(letters)-1
	if letters[right] <= target || target < letters[0] {
		return letters[0]
	}

	for left+1 < right {
		mid := left + (right-left)/2
		if letters[mid] <= target {
			left = mid
		} else {
			right = mid
		}
	}
	return letters[right]
}

//72. Search 2D Array
func searchMatrix(matrix [][]int, target int) bool {
	m := len(matrix)
	n := len(matrix[0])

	left, right := 0, n*m-1

	for left <= right {
		mid := (left + right) / 2
		midVal := matrix[mid/n][mid%n]

		if midVal == target {
			return true
		}
		if midVal > target {
			right = mid - 1
		}
		if midVal < target {
			left = mid + 1
		}
	}

	return false
}
