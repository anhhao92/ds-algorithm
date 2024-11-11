package problems

import (
	"slices"
)

// LC 252
func CanAttendMeeting(intervals [][]int) bool {
	slices.SortFunc(intervals, func(a, b []int) int {
		return a[0] - b[0]
	})
	for i := 1; i < len(intervals); i++ {
		prev := intervals[i-1]
		cur := intervals[i]
		// start < end
		if cur[0] < prev[1] {
			return false
		}
	}
	return true
}

// LC 253
func MinMeetingRooms(intervals [][]int) int {
	starts := make([]int, len(intervals))
	ends := make([]int, len(intervals))
	for i, v := range intervals {
		starts[i] = v[0]
		ends[i] = v[1]
	}
	slices.Sort(starts)
	slices.Sort(ends)
	res, count := 0, 0
	for i, j := 0, 0; i < len(intervals); {
		if starts[i] < ends[j] {
			i++
			count++
		} else {
			j++
			count--
		}
		res = max(res, count)
	}
	return res
}

// LC 56
func MergeInterval(intervals [][]int) [][]int {
	slices.SortFunc(intervals, func(a, b []int) int {
		return a[0] - b[0]
	})
	result := [][]int{intervals[0]}
	for i := 1; i < len(intervals); i++ {
		lastEnd := result[len(result)-1][1]
		if intervals[i][0] <= lastEnd {
			result[len(result)-1][1] = max(lastEnd, intervals[i][1])
		} else {
			result = append(result, intervals[i])
		}
	}
	return result
}

// LC 57
func InsertInterval(intervals [][]int, newInterval []int) [][]int {
	res := [][]int{}
	for i, interval := range intervals {
		// new interval < start
		if newInterval[1] < interval[0] {
			res = append(res, newInterval)
			res = append(res, intervals[i:]...)
			return res
		} else if newInterval[0] > interval[1] {
			res = append(res, interval)
		} else {
			newInterval[0] = min(newInterval[0], interval[0])
			newInterval[1] = max(newInterval[1], interval[1])
		}
	}
	res = append(res, newInterval)
	return res
}

// LC 435
func EraseOverlapIntervals(intervals [][]int) int {
	slices.SortFunc(intervals, func(a, b []int) int {
		return a[0] - b[0]
	})
	count := 0
	end := intervals[0][1]
	for i := 1; i < len(intervals); i++ {
		if end <= intervals[i][0] {
			end = intervals[i][1]
		} else {
			count++
		}
	}
	return count
}
