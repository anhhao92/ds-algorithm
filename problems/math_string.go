package problems

import (
	"math"
	"slices"
	"strings"
)

func Multiply(num1 string, num2 string) string {
	res := make([]byte, len(num1)+len(num2))
	b1, b2 := []byte(num1), []byte(num2)
	slices.Reverse(b1)
	slices.Reverse(b2)
	for i := 0; i < len(b1); i++ {
		for j := 0; j < len(b2); j++ {
			digit := (b1[i] - '0') * (b2[j] - '0')
			res[i+j] += digit
			res[i+j+1] += res[i+j] / 10
			res[i+j] = res[i+j] % 10
		}
	}
	slices.Reverse(res)
	if res[0] == 0 {
		res = res[1:]
	}
	for i := range res {
		res[i] += byte('0')
	}
	return string(res)
}

func longestCommonPrefix(strs []string) string {
	common := strs[0]
	for i := 1; i < len(strs); i++ {
		for !strings.HasPrefix(strs[i], common) {
			common = common[:len(common)-1]
		}
	}
	return common
}

func GroupAnagrams(strs []string) [][]string {
	hs := map[[26]byte][]string{}
	res := [][]string{}
	for _, s := range strs {
		key := [26]byte{}
		for _, c := range s {
			key[c-'a']++
		}
		hs[key] = append(hs[key], s)
	}
	for _, v := range hs {
		res = append(res, v)
	}
	return res
}

func setZeroes(matrix [][]int) {
	m, n := len(matrix), len(matrix[0])
	rowZero := false
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if matrix[i][j] == 0 {
				matrix[0][j] = 0
				if i > 0 {
					matrix[i][0] = 0
				} else {
					rowZero = true
				}
			}
		}
	}

	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			if matrix[0][j] == 0 || matrix[i][0] == 0 {
				matrix[i][j] = 0
			}
		}
	}
	// 1st col is zero
	if matrix[0][0] == 0 {
		for i := 0; i < m; i++ {
			matrix[i][0] = 0
		}
	}
	// 1st row is zero
	if rowZero {
		for j := 0; j < n; j++ {
			matrix[0][j] = 0
		}
	}
}

// LC7
func reverse(x int) int {
	result := 0
	for x != 0 {
		digit := x % 10
		if result > math.MaxInt32/10 || result == math.MaxInt32/10 && digit > math.MaxInt32%10 {
			return 0
		}
		if result < math.MinInt32/10 || result == math.MinInt32/10 && digit < math.MinInt32%10 {
			return 0
		}
		result = result*10 + digit
		x /= 10
	}
	return result
}
func spiralOrder(matrix [][]int) []int {
	top, down, left, right := 0, len(matrix), 0, len(matrix[0])
	res := make([]int, 0, len(matrix)*len(matrix[0]))
	for left < right && top < down {
		for i := left; i < right; i++ {
			res = append(res, matrix[top][i])
		}
		top++
		for i := top; i < down; i++ {
			res = append(res, matrix[i][right-1])
		}
		right--
		for i := right - 1; i >= left; i-- {
			res = append(res, matrix[down-1][i])
		}
		down--
		for i := down - 1; i >= top; i-- {
			res = append(res, matrix[i][left])
		}
		left++
	}
	return res
}
