package problems

import (
	"math"
	"slices"
)

func BinarySearch(nums []int, target int) int {
	l, r := 0, len(nums)
	for l < r {
		mid := l + (r-l)/2
		if target <= nums[mid] {
			r = mid
		} else {
			l = mid + 1
		}
	}
	return l
}

// LC 300
func FindLengthOfLIS(nums []int) int {
	// this is not actual LIS just the length of LIS
	lis := []int{}
	for i := 0; i < len(nums); i++ {
		pos := BinarySearch(lis, nums[i])
		if pos == len(lis) {
			lis = append(lis, nums[i])
		} else {
			lis[pos] = nums[i]
		}
	}
	return len(lis)
}

func ReconstructLIS(nums []int) []int {
	tailIndices := []int{}
	prevIndices := make([]int, len(nums))
	for i := range prevIndices {
		prevIndices[i] = -1
	}
	binarySearchIndices := func(indices []int, target int) int {
		l, r := 0, len(indices)
		for l < r {
			m := l + (r-l)/2
			midVal := nums[indices[m]]
			if target <= midVal {
				r = m
			} else {
				l = m + 1
			}
		}
		return l
	}
	for i := 0; i < len(nums); i++ {
		pos := binarySearchIndices(tailIndices, nums[i])
		if pos == len(tailIndices) {
			tailIndices = append(tailIndices, i)
		} else {
			tailIndices[pos] = i
		}

		if pos > 0 {
			prevIndices[i] = tailIndices[pos-1]
		}
	}
	res := []int{}
	for i := tailIndices[len(tailIndices)-1]; i > -1; i = prevIndices[i] {
		res = append(res, nums[i])
	}
	slices.Reverse(res)
	return res
}

// LC 33 Find value in rotated sorted array
func SearchSortedArray(nums []int, target int) int {
	left, right := 0, len(nums)-1

	for left <= right {
		mid := left + (right-left)/2
		if nums[mid] == target {
			return mid
		}
		if nums[left] <= nums[mid] {
			if nums[left] <= target && target < nums[mid] {
				right = mid - 1
			} else {
				left = mid + 1
			}
		} else {
			if nums[mid] < target && target <= nums[right] {
				left = mid + 1
			} else {
				right = mid - 1
			}
		}
	}
	return -1
}

// LC 81
func SearchSortedArrayII(nums []int, target int) bool {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)/2
		if nums[mid] == target {
			return true
		}
		// we know range [l,m] or [m,r] increasing order
		if nums[left] < nums[mid] {
			if nums[left] <= target && target < nums[mid] {
				right = mid - 1
			} else {
				left = mid + 1
			}
		} else if nums[left] > nums[mid] {
			if nums[mid] < target && target <= nums[right] {
				left = mid + 1
			} else {
				right = mid - 1
			}
		} else {
			left++
		}
	}
	return false
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

// LC 34
func SearchRange(nums []int, target int) []int {
	modifiedBinarySearch := func(leftMost bool) int {
		l, r := 0, len(nums)-1
		res := -1
		for l <= r {
			mid := l + (r-l)/2
			if nums[mid] > target {
				r = mid - 1
			} else if nums[mid] < target {
				l = mid + 1
			} else {
				res = mid
				if leftMost {
					r = mid - 1
				} else {
					l = mid + 1
				}
			}
		}
		return res
	}
	start, end := modifiedBinarySearch(true), modifiedBinarySearch(false)
	return []int{start, end}
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

// 72. Search 2D Array
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

/*
LC 4
[x l1 | r1 x x x]
[x x x l2 | r2 x]
Find the partitions satify the condition l1 <= r2 && l2 <= r1
*/
func FindMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	m, n := len(nums1), len(nums2)
	if m > n {
		return FindMedianSortedArrays(nums2, nums1)
	}
	l, r := 0, m
	for l <= r {
		mid1 := (l + r) / 2
		mid2 := (m+n+1)/2 - mid1
		l1, r1 := math.MinInt32, math.MaxInt32
		if mid1 < m {
			r1 = nums1[mid1]
		}
		if mid1-1 >= 0 {
			l1 = nums1[mid1-1]
		}
		l2, r2 := math.MinInt32, math.MaxInt32
		if mid2 < n {
			r2 = nums2[mid2]
		}
		if mid2-1 >= 0 {
			l2 = nums2[mid2-1]
		}
		if l1 <= r2 && l2 <= r1 {
			if (m+n)%2 == 0 {
				return float64(max(l1, l2)+min(r1, r2)) / 2
			}
			return float64(max(l1, l2))
		} else if l1 > r2 { // too far the right
			r = mid1 - 1
		} else {
			l = mid1 + 1
		}
	}
	return 0
}

// LC 540
func singleNonDuplicate(nums []int) int {
	n := len(nums)
	l, r := 0, n-1
	for l <= r {
		mid := l + (r-l)/2
		if (mid-1 < 0 || nums[mid] != nums[mid-1]) && (mid+1 == n || nums[mid] != nums[mid+1]) {
			return nums[mid]
		}
		left := mid
		if mid-1 >= 0 && nums[mid] == nums[mid-1] {
			left = mid - 1
		}
		if left%2 == 0 {
			l = mid + 1
		} else {
			r = mid - 1
		}
	}
	return -1
}

// LCC 162/852
func findPeakElement(nums []int) int {
	l, r := 0, len(nums)-1
	for l < r {
		mid := l + (r-l)/2
		if nums[mid] > nums[mid+1] {
			r = mid
		} else {
			l = mid + 1
		}
	}
	// for l <= r {
	//     mid := l + (r - l) / 2
	//     if mid > 0 && arr[mid - 1] > arr[mid] {
	//         r = mid - 1
	//     } else if mid < len(arr) -1 && arr[mid] < arr[mid+1] {
	//         l = mid + 1
	//     } else {
	//         return mid
	//     }
	// }
	return l
}

// LC 2300
func successfulPairs(spells []int, potions []int, success int64) []int {
	slices.Sort(potions)
	res := []int{}
	for _, s := range spells {
		l, r := 0, len(potions)-1
		index := len(potions)
		for l <= r {
			mid := l + (r-l)/2
			if int64(s*potions[mid]) >= success {
				r = mid - 1
				index = mid
			} else {
				l = mid + 1
			}
		}
		res = append(res, len(potions)-index)
	}
	return res
}

// LC 1011
func shipWithinDays(weights []int, days int) int {
	l, r := 0, 0
	for _, v := range weights {
		l = max(l, v)
		r += v
	}
	canFitShip := func(maxCap int) bool {
		currentCap, count := 0, 1
		for _, v := range weights {
			currentCap += v
			if currentCap > maxCap {
				currentCap = v
				count++
			}
		}
		return count <= days
	}
	res := 0
	for l <= r {
		mid := l + (r-l)/2
		if canFitShip(mid) {
			res = mid
			r = mid - 1
		} else {
			l = mid + 1
		}
	}
	return res
}

// LC 2616
func minimizeMax(nums []int, p int) int {
	if p < 1 {
		return 0
	}
	slices.Sort(nums)
	l, r := 0, nums[len(nums)-1]-nums[0]
	isValidPair := func(threshold int) bool {
		count := 0
		for i := 1; i < len(nums); i++ {
			if nums[i]-nums[i-1] <= threshold {
				count++
				i++
			}
			if count == p {
				return true
			}

		}
		return false
	}
	for l < r {
		mid := l + (r-l)/2
		if isValidPair(mid) {
			r = mid
		} else {
			l = mid + 1
		}
	}
	return l
}

// LC 981
type (
	TimeValue struct {
		value string
		time  int
	}
	TimeMap map[string][]TimeValue
)

func NewTimeMap() TimeMap {
	return TimeMap{}
}

func (this *TimeMap) Set(key string, value string, timestamp int) {
	val, ok := (*this)[key]
	if ok {
		val = append(val, TimeValue{value: value, time: timestamp})
		(*this)[key] = val
	} else {
		(*this)[key] = []TimeValue{{value: value, time: timestamp}}
	}
}

func (this *TimeMap) Get(key string, timestamp int) string {
	values := (*this)[key]
	l, r := 0, len(values)-1
	res := ""
	for l <= r {
		mid := l + (r-l)/2
		val := values[mid]
		if val.time == timestamp {
			return val.value
		} else if val.time < timestamp {
			res = val.value
			l = mid + 1
		} else {
			r = mid - 1
		}
	}
	return res
}

// LC 1898
func maximumRemovals(s string, p string, removable []int) int {
	isSubSequence := func(k int) bool {
		removed := make(map[int]bool)
		for i := 0; i <= k; i++ {
			removed[removable[i]] = true
		}

		p1, p2 := 0, 0
		for p1 < len(s) && p2 < len(p) {
			if removed[p1] || s[p1] != p[p2] {
				p1++
			} else {
				p1++
				p2++
			}
		}
		return p2 == len(p)
	}

	l, r := 0, len(removable)-1
	res := 0
	for l <= r {
		mid := l + (r-l)/2
		if isSubSequence(mid) {
			res = max(res, mid+1)
			l = mid + 1
		} else {
			r = mid - 1
		}
	}
	return res
}

// LC 410
func splitArray(nums []int, k int) int {
	l, r := 0, 0 // max, sum
	for _, v := range nums {
		l = max(l, v)
		r += v
	}
	res := r
	canSplit := func(maxSum int) bool {
		curSum := 0
		subArray := 1
		// count how many sub arrays <= maxSum
		for _, v := range nums {
			curSum += v
			if curSum > maxSum {
				subArray++
				curSum = v
			}
		}
		// less than it means maxSum is too big
		return subArray <= k
	}
	for l <= r {
		mid := l + (r-l)/2
		if canSplit(mid) {
			r = mid - 1
			// mid can be the answer
			res = mid
		} else {
			l = mid + 1
		}
	}
	return res
}

// LC 1268
func SearchSuggestedProducts(products []string, searchWord string) [][]string {
	slices.Sort(products)
	result := [][]string{}
	start, end := 0, len(products)-1
	for i := 0; i < len(searchWord); i++ {
		character := searchWord[i]
		// if word doesn't match or length out of bound we should eliminate it
		for start <= end && (i >= len(products[start]) || products[start][i] != character) {
			start++
		}
		for start <= end && (i >= len(products[end]) || products[end][i] != character) {
			end--
		}
		result = append(result, products[start:start+min(3, end-start+1)])
	}
	return result
}

// LC 3269
func MinNumberOfSeconds(mountainHeight int, workerTimes []int) int64 {
	canComplete := func(maxTime int) bool {
		reducedHeight := 0
		for _, time := range workerTimes {
			l, r := 1, mountainHeight
			for l <= r {
				mid := (l + r) / 2
				cost := time * mid * (mid + 1) / 2 // time * sum(1..n)
				if cost <= maxTime {
					l = mid + 1
				} else {
					r = mid - 1
				}
			}
			reducedHeight += r
			if reducedHeight >= mountainHeight {
				return true
			}
		}
		return reducedHeight >= mountainHeight
	}
	minWorker := slices.Min(workerTimes)
	l, r := 1, minWorker*mountainHeight*(mountainHeight+1)/2
	for l < r {
		mid := l + (r-l)/2
		if canComplete(mid) {
			r = mid
		} else {
			l = mid + 1
		}
	}
	return int64(l)
}
