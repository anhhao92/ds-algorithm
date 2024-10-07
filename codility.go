package main

import (
	"math"
	"slices"
	"strconv"
	"strings"
)

// incomplete
func MaxSumPairEven(nums []int) int {
	maxSum, n := math.MinInt32, len(nums)
	prev, count := n-1, 0
	for i := 0; i < n; i++ {
		sum := nums[i] + nums[prev]
		if sum%2 == 0 {
			if sum > maxSum {
				count = 1
				maxSum = sum
			} else if sum == maxSum {
				count++
			}
		}
		prev = i
	}
	return count
}

func MinLengthFixedBoard(nums []int) int {
	if len(nums) < 2 {
		return 1
	}
	slices.Sort(nums)
	left, right := 1, nums[len(nums)-1]-nums[0]
	result := 1
	countNumberOfBoard := func(nums []int, boardLen int) int {
		prev := nums[0]
		count := 1
		for i := 1; i < len(nums); i++ {
			if nums[i]-prev > boardLen {
				prev = nums[i]
				count++
			}
		}
		return count
	}
	for left <= right {
		mid := left + (right-left)/2
		if countNumberOfBoard(nums, mid) <= 2 {
			result = mid
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	return result
}

func minimumDeletions(s string) int {
	count := 0
	res := 0
	for i := 0; i < len(s); i++ {
		if s[i] == 'b' {
			count++
		} else {
			res = min(res+1, count)
		}
	}
	return res
}

func countCarPassingSpeedCam(s string) int {
	result, n := 0, len(s)
	for i, count := 0, 0; i < n; i++ {
		if s[i] == '.' {
			count++
		} else if s[i] == '<' {
			result += count
		}
	}
	for i, count := n-1, 0; i >= 0; i-- {
		if s[i] == '.' {
			count++
		} else if s[i] == '>' {
			result += count
		}
	}
	return result
}

func GameRound(a string, b string) int {
	getTimeInMinutes := func(t string) int {
		arr := strings.Split(t, ":")
		hour, _ := strconv.Atoi(arr[0])
		minute, _ := strconv.Atoi(arr[1])
		return hour*60 + minute
	}
	start := getTimeInMinutes(a)
	end := getTimeInMinutes(b)
	rounds := 0

	if end < start {
		end += 24 * 60
	}

	for i := start + 15 - (start % 15); i <= end; i += 15 {
		rounds++
	}

	return rounds
}

func MaxSumNonAttackRooks(board [][]int) int {
	rowLen, colLen := len(board), len(board[0])
	row := make([][2]int, rowLen)
	col := make([][2]int, colLen)

	for i := 0; i < rowLen; i++ {
		row1, row2 := 0, 0
		col1, col2 := 0, 0
		for j := 0; j < colLen; j++ {
			if board[i][j] > row1 {
				row2 = row1
				row1 = board[i][j]
				col2 = col1
				col1 = j
			} else if board[i][j] > row2 {
				row2 = board[i][j]
				col2 = j
			}
		}
		row[i][0] = col1
		row[i][1] = col2
	}

	for j := 0; j < colLen; j++ {
		col1, col2 := 0, 0
		row1, row2 := 0, 0
		for i := 0; i < rowLen; i++ {
			if board[i][j] > col1 {
				col2 = col1
				col1 = board[i][j]
				row2 = row1
				row1 = i
			} else if board[i][j] > col2 {
				col2 = board[i][j]
				row2 = i
			}
		}
		col[j][0] = row1
		col[j][1] = row2
	}

	result := 0
	for i := 0; i < rowLen; i++ {
		c := row[i][0]
		for j := 0; j < colLen; j++ {
			if j != c {
				row1, row2 := col[j][0], col[j][1]
				if col[j][0] != i {
					result = max(result, board[i][c]+board[row1][j])
				} else {
					result = max(result, board[i][c]+board[row2][j])
				}
			}
		}
	}

	for j := 0; j < colLen; j++ {
		r := col[j][0]
		for i := 0; i < rowLen; i++ {
			if i != r {
				col1, col2 := col[j][0], col[j][1]
				if col[j][0] != i {
					result = max(result, board[r][j]+board[i][col1])
				} else {
					result = max(result, board[r][j]+board[i][col2])
				}
			}
		}
	}
	return result
}

func countEnemyInsideFlashlightRange(direction string, radius int, x, y []int) int {
	leftBoundary, rightBoundary := [2]float64{0, 0}, [2]float64{0, 0}
	count := 0
	switch direction {
	case "U":
		rightBoundary = [2]float64{45, 90}
		leftBoundary = [2]float64{90, 135}
	case "D":
		rightBoundary = [2]float64{225, 270}
		leftBoundary = [2]float64{270, 315}
	case "L":
		rightBoundary = [2]float64{135, 180}
		leftBoundary = [2]float64{180, 225}
	case "R":
		rightBoundary = [2]float64{315, 360}
		leftBoundary = [2]float64{0, 45}
	}

	for i := 0; i < len(x); i++ {
		dist := x[i]*x[i] + y[i]*y[i]
		if dist <= radius*radius {
			angular := math.Atan2(float64(y[i]), float64(x[i])) * 180 / math.Pi
			if angular < 0 {
				angular += 360
			}
			if (angular >= leftBoundary[0] && angular <= leftBoundary[1]) || (angular >= rightBoundary[0] && angular <= rightBoundary[1]) {
				count++
			}
		}
	}
	return count
}
func CanFormSquare(a, b int) int {
	l, r := 1, a+b/4
	res := 0
	for l <= r {
		mid := (r + l) / 2
		if a/mid+b/mid >= 4 {
			res = mid
			l = mid + 1
		} else {
			r = mid - 1
		}
	}
	return res
}

func CountBattleShip(b []string) []int {
	result := []int{0, 0, 0}
	board := make([][]byte, len(b))
	for i, s := range b {
		board[i] = []byte(s)
	}
	lenRow, lenCol := len(board), len(board[0])
	var dfs func(r, c int) int
	dfs = func(r, c int) int {
		shipSize := 1
		board[r][c] = '.'
		directions := [4][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
		for _, d := range directions {
			dr, dc := d[0]+r, d[1]+c
			if dr >= 0 && dr < lenRow && dc >= 0 && dc < lenCol && board[dr][dc] == '#' {
				shipSize += dfs(dr, dc)
			}
		}
		return shipSize
	}
	for i := 0; i < lenRow; i++ {
		for j := 0; j < lenCol; j++ {
			if board[i][j] == '#' {
				kind := dfs(i, j) - 1
				result[kind]++
			}
		}
	}
	return result
}

func GenerateString(n int) string {
	res := make([]byte, n)
	repeat := 1
	if n > 26 {
		if n%26 == 0 {
			repeat = n / 26
		} else if n%2 == 0 {
			repeat = repeat + n/26
		} else {
			repeat = n
		}
	}
	for i, k := byte('a'), 0; i <= byte('z') && k < n; i++ {
		for j := 0; j < repeat; j++ {
			res[k] = i
			k++
		}
	}
	return string(res)
}
