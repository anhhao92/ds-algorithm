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
func ReverseInteger(x int) int {
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

// LC 54
func SpiralOrder(matrix [][]int) []int {
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

// LC 59
func GenerateSpiralMatrix(n int) [][]int {
	top, down, left, right := 0, n-1, 0, n-1
	res := make([][]int, n)
	for i := range n {
		res[i] = make([]int, n)
	}
	j := 1
	for left < right && top < down {
		for i := left; i < right; i++ {
			res[top][i] = j
			j++
		}
		top++
		for i := top; i < down; i++ {
			res[i][right] = j
			j++
		}
		right--
		for i := right; i >= left; i-- {
			res[down][i] = j
			j++
		}
		down--
		for i := down; i >= top; i-- {
			res[i][left] = j
			j++
		}
	}
	return res
}

// LC 48
func rotateMatrix(matrix [][]int) {
	l, r := 0, len(matrix)-1
	for l < r {
		for i := 0; i < r-l; i++ {
			top, bottom := l, r
			topLeft := matrix[top][l+i]
			matrix[top][l+i] = matrix[bottom-i][l]
			matrix[bottom-i][l] = matrix[bottom][r-i]
			matrix[bottom][r-i] = matrix[top+i][r]
			matrix[top+i][r] = topLeft
		}
		l++
		r--
	}
}

// LC 67
func addBinary(a string, b string) string {
	carry := byte(0)
	res := []byte{}
	A, B := []byte(a), []byte(b)
	slices.Reverse(A)
	slices.Reverse(B)
	for i := 0; i < max(len(A), len(B)); i++ {
		bitA, bitB := byte(0), byte(0)
		if i < len(a) {
			bitA = A[i] - byte('0')
		}
		if i < len(b) {
			bitB = B[i] - byte('0')
		}
		total := bitA + bitB + carry
		carry = total / 2
		total = total % 2
		res = append(res, total+'0')
	}
	if carry > 0 {
		res = append(res, '1')
	}
	slices.Reverse(res)
	return string(res)
}

// LC 989
func addToArrayForm(num []int, k int) []int {
	slices.Reverse(num)
	for i := 0; k > 0; i++ {
		if i < len(num) {
			num[i] += k % 10
		} else {
			num = append(num, k%10)
		}
		carry := num[i] / 10
		num[i] = num[i] % 10
		k = k / 10
		k += carry
	}
	slices.Reverse(num)
	return num
}

// LC 440
func FindKthNumber(n int, k int) int {
	cur := 1
	count := func(cur int) int {
		res := 0
		neighbor := cur + 1
		for cur <= n {
			res += min(neighbor, n+1) - cur
			cur *= 10
			neighbor *= 10
		}
		return res
	}

	for i := 1; i < k; {
		steps := count(cur)
		if i+steps <= k {
			i += steps
			cur++
		} else {
			cur *= 10
			i++
		}
	}
	return cur
}

// LC 12
func IntToRoman(num int) string {
	symbols := []struct {
		letter string
		value  int
	}{
		{"I", 1},
		{"IV", 4},
		{"V", 5},
		{"IX", 9},
		{"X", 10},
		{"XL", 40},
		{"L", 50},
		{"XC", 90},
		{"C", 100},
		{"CD", 400},
		{"D", 500},
		{"CM", 900},
		{"M", 1000},
	}
	var sb strings.Builder
	for i := len(symbols) - 1; i >= 0; i-- {
		s, v := symbols[i].letter, symbols[i].value
		if num/v > 0 {
			sb.WriteString(strings.Repeat(s, num/v))
			num %= v
		}
	}
	return sb.String()
}

// LC 13
func RomanToInt(s string) int {
	res := 0
	n := len(s)
	symbols := map[byte]int{
		'I': 1,
		'V': 5,
		'X': 10,
		'L': 50,
		'C': 100,
		'D': 500,
		'M': 1000,
	}
	for i := 0; i < n; i++ {
		if i+1 < n && symbols[s[i]] < symbols[s[i+1]] {
			res += -symbols[s[i]]
		} else {
			res += symbols[s[i]]
		}
	}
	return res
}

// LC 273
func IntegerNumberToWords(num int) string {
	if num == 0 {
		return "Zero"
	}
	hashMap := map[int]string{
		90: "Ninety",
		80: "Eighty",
		70: "Seventy",
		60: "Sixty",
		50: "Fifty",
		40: "Forty",
		30: "Thirty",
		20: "Twenty",
		19: "Nineteen",
		18: "Eighteen",
		17: "Seventeen",
		16: "Sixteen",
		15: "Fifteen",
		14: "Fourteen",
		13: "Thirteen",
		12: "Twelve",
		11: "Eleven",
		10: "Ten",
		9:  "Nine",
		8:  "Eight",
		7:  "Seven",
		6:  "Six",
		5:  "Five",
		4:  "Four",
		3:  "Three",
		2:  "Two",
		1:  "One",
	}
	res := []string{}
	getStr := func(n int) string {
		res := []string{}
		h := n / 100
		if h > 0 {
			res = append(res, hashMap[h]+" Hundred")
		}
		l := n % 100
		if l >= 20 {
			tens, ones := l/10, l%10
			res = append(res, hashMap[tens*10])
			if ones > 0 {
				res = append(res, hashMap[ones])
			}
		} else if l > 0 {
			res = append(res, hashMap[l])
		}
		return strings.Join(res, " ")
	}
	suffix := [5]string{"", " Thousand", " Million", " Billion"}
	for i := 0; i < 5 && num > 0; i++ {
		digit := num % 1000
		s := getStr(digit)
		if s != "" {
			res = append(res, s+suffix[i])
		}
		num /= 1000
	}
	slices.Reverse(res)
	return strings.Join(res, " ")
}

// LC 149
func maxPoints(points [][]int) int {
	res := 1
	for i := range points {
		hashMap := map[float64]int{}
		x1, y1 := points[i][0], points[i][1]
		for j := i + 1; j < len(points); j++ {
			x2, y2 := points[j][0], points[j][1]
			slope := math.MaxFloat32
			if y2 != y1 {
				slope = float64(x2-x1) / float64(y2-y1)
			}
			hashMap[slope]++
			res = max(res, hashMap[slope]+1)
		}
	}
	return res
}

// LC 1071
func GreatestCommonDivisorOfStrings(str1 string, str2 string) string {
	gcd := func(x, y int) int {
		for y != 0 {
			x, y = y, x%y
		}
		return x
	}

	if str1+str2 == str2+str1 {
		return str1[:gcd(len(str1), len(str2))]
	}
	return ""
}

// 204
func countPrimes(n int) int {
	if n < 2 {
		return 0
	}
	prime := make([]bool, n)
	for i := range prime {
		prime[i] = true
	}
	for i := 2; i*i < n; i++ {
		if prime[i] {
			for j := i * i; j < n; j += i {
				prime[j] = false
			}
		}
	}
	count := 0
	for i := 2; i < n; i++ {
		if prime[i] {
			count++
		}
	}
	return count
}

// LC 6
func convertZigZag(s string, numRows int) string {
	if numRows == 1 {
		return s
	}
	var sb strings.Builder
	for r := range numRows {
		inc := 2 * (numRows - 1)
		for i := r; i < len(s); i += inc {
			sb.WriteByte(s[i])
			if r > 0 && r < numRows-1 && i+inc-2*r < len(s) {
				sb.WriteByte(s[i+inc-2*r])
			}
		}
	}
	return sb.String()
}

// L 1727
func largestSubmatrix(matrix [][]int) int {
	rows, cols := len(matrix), len(matrix[0])
	prev := make([]int, cols)
	res := 0
	for i := 0; i < rows; i++ {
		heights := matrix[i]
		for j := 0; j < cols; j++ {
			if heights[j] > 0 {
				heights[j] += prev[j]
			}
		}
		h := slices.Clone(heights)
		slices.SortFunc(h, func(a, b int) int { return b - a })
		for r := 0; r < cols; r++ {
			res = max(res, (r+1)*h[r])
		}
		prev = heights
	}

	return res
}

// LC 1041
func isRobotBounded(instructions string) bool {
	dirX, dirY := 0, 1
	x, y := 0, 0
	for _, d := range instructions {
		if d == 'G' {
			x, y = x+dirX, y+dirY
		} else if d == 'L' {
			dirX, dirY = -dirY, dirX
		} else {
			dirX, dirY = dirY, -dirX
		}
	}
	// same postion or change direction
	return (x == 0 && y == 0) || (dirX != 0 && dirY != 1)
}
