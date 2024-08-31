package problems

import (
	"strconv"
	"strings"
)

type Stack[T any] struct {
	size int
	val  map[int]T
}

func NewStack[T any]() *Stack[T] {
	return &Stack[T]{val: make(map[int]T)}
}

func (s *Stack[T]) Push(v T) {
	s.val[s.size] = v
	s.size++
}

func (s *Stack[T]) Pop() interface{} {
	if s.size-1 < 0 {
		return nil
	}
	res := s.val[s.size-1]
	delete(s.val, s.size-1)
	s.size--

	return res
}

func (s *Stack[T]) Values() []T {
	res := make([]T, s.size)
	for i := 0; i < s.size; i++ {
		res[i] = s.val[i]
	}
	return res
}

func isNumber(s byte) bool {
	return s >= '0' && s <= '9'
}

func calculate(s string) int {
	operator := byte('+')
	prev := 0
	sum := 0

	for i := 0; i < len(s); i++ {
		if s[i] == ' ' {
			continue
		}
		// digit
		if isNumber(s[i]) {
			num := 0
			for i < len(s) && isNumber(s[i]) {
				num = 10*num + int(s[i]-'0')
				i++
			}
			i--
			// operator
			switch operator {
			case '+':
				prev = num
				sum += num
			case '-':
				prev = -num
				sum -= num
			case '*':
				sum -= prev
				sum += prev * num
				prev = prev * num
			case '/':
				sum -= prev
				sum += prev / num
				prev = prev / num
			}

		} else {
			operator = s[i]
		}
	}
	return sum
}

// LC20
func isValidParentheses(s string) bool {
	stack := []byte{}
	closeOpen := map[byte]byte{
		')': '(',
		']': '[',
		'}': '{',
	}
	for i := 0; i < len(s); i++ {
		if v, ok := closeOpen[s[i]]; ok {
			if len(stack) > 0 && v == stack[len(stack)-1] {
				stack = stack[:len(stack)-1]
			} else {
				return false
			}
		} else {
			stack = append(stack, s[i])
		}
	}
	return len(stack) == 0
}

func calculateParentheses(s string) int {
	result, number, sign := 0, 0, 1
	stack := []int{}

	for _, c := range s {
		if c >= '0' && c <= '9' {
			number = (number * 10) + int(c-'0')
		}
		if c == '+' {
			result += sign * number
			sign = 1
			number = 0
		}
		if c == '-' {
			result += sign * number
			sign = -1
			number = 0
		}
		// store current result + sign into stack then recalculate from ()
		if c == '(' {
			stack = append(stack, result)
			stack = append(stack, sign)

			sign = 1
			result = 0
		}
		if c == ')' {
			result += sign * number
			number = 0

			result *= stack[len(stack)-1]
			result += stack[len(stack)-2]
			stack = stack[:len(stack)-2]
		}
	}

	if number != 0 {
		result += sign * number
	}

	return result
}

// wrong
func CalculateComplex(s string) int {
	operator := byte('+')
	prev := 0
	sum := 0
	stack := []int{}

	for i := 0; i < len(s); i++ {
		if s[i] == ' ' {
			continue
		}
		// digit
		if isNumber(s[i]) {
			num := 0
			for i < len(s) && isNumber(s[i]) {
				num = 10*num + int(s[i]-'0')
				i++
			}
			i--
			// operator
			switch operator {
			case '+':
				prev = num
				sum += num
			case '-':
				prev = -num
				sum -= num
			case '*':
				sum -= prev
				sum += prev * num
				prev = prev * num
			case '/':
				sum -= prev
				sum += prev / num
				prev = prev / num
			}

		} else if s[i] == '(' {
			stack = append(stack, sum)
			stack = append(stack, int(operator))
			// reset sum & operator
			sum = 0
			// prev = 0
			// operator = '+'
		} else if s[i] == ')' {
			operator = byte(stack[len(stack)-1])
			switch operator {
			case '+':
				sum = stack[len(stack)-2] + sum
			case '-':
				sum = stack[len(stack)-2] - sum
			case '*':
				sum = stack[len(stack)-2] * sum
			case '/':
				sum = stack[len(stack)-2] / sum
			}
			stack = stack[:len(stack)-2]
		} else {
			operator = s[i]
		}
	}
	return sum

}

func dailyTemperatures(temperatures []int) []int {
	table := map[int]int{}
	res := make([]int, len(temperatures))
	stack := []int{}
	for i := range temperatures {
		for len(stack) > 0 {
			index := stack[len(stack)-1]
			if temperatures[i] > temperatures[index] {
				table[index] = i - index
				stack = stack[:len(stack)-1]
				res[index] = table[index]
			} else {
				break
			}
		}
		stack = append(stack, i)
	}
	return res
}

func nextGreaterElementI(nums1 []int, nums2 []int) []int {
	table := map[int]int{}
	res := make([]int, len(nums1))
	stack := []int{}
	for i := range nums2 {
		for len(stack) > 0 {
			topElement := stack[len(stack)-1]
			if nums2[i] > topElement {
				table[topElement] = nums2[i]
				stack = stack[:len(stack)-1]
				//res[index] = table[index]
			} else {
				break
			}
		}
		stack = append(stack, i)
	}
	for len(stack) > 0 {
		topElement := stack[len(stack)-1]
		table[topElement] = -1
	}
	for i := range res {
		res[i] = table[nums1[i]]
	}
	return res
}

func nextGreaterElements(nums []int) []int {
	res := make([]int, len(nums))
	stack := []int{}
	for i := range res {
		res[i] = -1
	}
	for i := range nums {
		for len(stack) > 0 {
			index := stack[len(stack)-1]
			if nums[i] > nums[index] {
				stack = stack[:len(stack)-1]
				res[index] = nums[i]
			} else {
				break
			}
		}
		stack = append(stack, i)
	}

	for i := range nums {
		for len(stack) > 0 {
			index := stack[len(stack)-1]
			if nums[i] > nums[index] {
				stack = stack[:len(stack)-1]
				res[index] = nums[i]
			} else {
				break
			}
		}
		if len(stack) == 0 {
			break
		}
	}
	return res
}

/**
 * Your MinStack object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Push(val);
 * obj.Pop();
 * param_3 := obj.Top();
 * param_4 := obj.GetMin();
 */
type MinStack struct {
	data     []int
	minStack []int
}

func NewMinStack() MinStack {
	return MinStack{data: []int{}, minStack: []int{}}
}

func (this *MinStack) Push(val int) {
	n := len(this.minStack)
	if n == 0 || val > this.minStack[n-1] {
		this.minStack = append(this.minStack, val)
	} else {
		topVal := this.minStack[n-1]
		this.minStack = append(this.minStack, topVal)
	}
	this.data = append(this.data, val)
}

func (this *MinStack) Pop() {
	n := len(this.data)
	this.data = this.data[:n-1]
	this.minStack = this.minStack[:n-1]
}

func (this *MinStack) Top() int {
	n := len(this.data)
	return this.data[n-1]
}

func (this *MinStack) GetMin() int {
	n := len(this.data)
	return this.minStack[n-1]
}

// LC150
func evalRPN(tokens []string) int {
	stack := []int{}
	for _, v := range tokens {
		n := len(stack)
		switch v {
		case "+":
			b, a := stack[n-1], stack[n-2]
			stack[n-2] = a + b
			stack = stack[:n-1]
		case "-":
			b, a := stack[n-1], stack[n-2]
			stack[n-2] = a - b
			stack = stack[:n-1]
		case "*":
			b, a := stack[n-1], stack[n-2]
			stack[n-2] = a * b
			stack = stack[:n-1]
		case "/":
			b, a := stack[n-1], stack[n-2]
			stack[n-2] = a / b
			stack = stack[:n-1]
		default:
			val, _ := strconv.Atoi(v)
			stack = append(stack, val)
		}
	}
	return stack[0]
}

// LC84
func largestRectangleArea(heights []int) int {
	maxArea := 0
	stack := [][2]int{}
	for i, h := range heights {
		start := i
		for len(stack) > 0 && stack[len(stack)-1][1] > h {
			index, height := stack[len(stack)-1][0], stack[len(stack)-1][1]
			maxArea = max(maxArea, (i-index)*height)
			start = index
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, [2]int{start, h})
	}
	for _, v := range stack {
		index, height := v[0], v[1]
		maxArea = max(maxArea, (len(heights)-index)*height)
	}
	return maxArea
}

// LC 2390
func removeStars(s string) string {
	stack := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		if s[i] == '*' {
			stack = stack[:len(stack)-1]
		} else {
			stack = append(stack, s[i])
		}
	}
	return string(stack)
}

// LC 946
func validateStackSequences(pushed []int, popped []int) bool {
	i := 0
	stack := []int{}
	for _, v := range pushed {
		stack = append(stack, v)
		for i < len(popped) && len(stack) > 0 && stack[len(stack)-1] == popped[i] {
			stack = stack[:len(stack)-1]
			i++
		}
	}
	return len(stack) == 0
}

// LC907
func sumSubarrayMins(arr []int) int {
	const MOD = int(1e9 + 7)
	total, n := 0, len(arr)
	stack := []int{}
	for i := 0; i < n; i++ {
		for len(stack) > 0 && arr[i] < arr[stack[len(stack)-1]] {
			j := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			left := j + 1
			if len(stack) > 0 {
				left = j - stack[len(stack)-1]
			}
			right := i - j
			total += arr[j] * right * left
		}
		stack = append(stack, i)
	}
	for i, j := range stack {
		left := j + 1
		if i > 0 {
			left = j - stack[i-1]
		}
		right := n - j
		total += arr[j] * right * left

	}
	return total % MOD
}

// LC1209
func removeAllAdjacentDuplicates(s string, k int) string {
	stack := [][2]int{} // [character, count]
	res := []byte{}
	for i := 0; i < len(s); i++ {
		n := len(stack)
		if n > 0 && byte(stack[n-1][0]) == s[i] {
			stack[n-1][1]++
			if stack[n-1][1] == k {
				stack = stack[:n-1]
			}
		} else {
			stack = append(stack, [2]int{int(s[i]), 1})
		}
	}
	for _, v := range stack {
		ch, cnt := byte(v[0]), v[1]
		for i := 0; i < cnt; i++ {
			res = append(res, ch)
		}
	}
	return string(res)
}

// LC 402 monotonic stack
func removeKdigits(num string, k int) string {
	stack := []byte{}
	for i := 0; i < len(num); i++ {
		for len(stack) > 0 && k > 0 && stack[len(stack)-1] > num[i] {
			stack = stack[:len(stack)-1]
			k--
		}
		stack = append(stack, num[i])
	}
	stack = stack[:len(stack)-k]
	for len(stack) > 0 && stack[0] == byte('0') {
		stack = stack[1:]
	}
	if len(stack) == 0 {
		return "0"
	}
	return string(stack)
}

// LC456
func find132pattern(nums []int) bool {
	stack := [][2]int{} // [num, minLeft]
	currentMin := nums[0]
	for i := 1; i < len(nums); i++ {
		for len(stack) > 0 && nums[i] >= stack[len(stack)-1][0] {
			stack = stack[:len(stack)-1]
		}
		if len(stack) > 0 && nums[i] > stack[len(stack)-1][1] {
			return true
		}
		stack = append(stack, [2]int{nums[i], currentMin})
		currentMin = min(currentMin, nums[i])
	}
	return false
}

// LC 394
func decodeString(s string) string {
	numStack := []int{}
	strStack := []string{}
	curNum, curStr := "", ""
	for i := 0; i < len(s); i++ {
		if s[i] >= '0' && s[i] <= '9' {
			curNum += string(s[i])
		} else if s[i] == '[' {
			num, _ := strconv.Atoi(curNum)
			numStack = append(numStack, num)
			strStack = append(strStack, curStr)
			curNum, curStr = "", ""
		} else if s[i] == ']' {
			curStr = strStack[len(strStack)-1] + strings.Repeat(curStr, numStack[len(numStack)-1])
			strStack = strStack[:len(strStack)-1]
			numStack = numStack[:len(numStack)-1]
		} else {
			curStr += string(s[i])
		}
	}
	return curStr
}

// LC71
func SimplifyPath(path string) string {
	stack := []string{}
	paths := strings.Split(path, "/")
	for _, current := range paths {
		if current == "." || current == "" {
			continue
		}
		if current == ".." {
			if len(stack) > 0 {
				stack = stack[:len(stack)-1]
			}
		} else {
			stack = append(stack, current)
		}
	}
	return "/" + strings.Join(stack, "/")
}

// LC895
type FreqStack struct {
	freqMap map[int]int
	bucket  map[int][]int
	maxFreq int
}

func NewFreqStack() FreqStack {
	return FreqStack{
		freqMap: make(map[int]int),
		bucket:  map[int][]int{},
		maxFreq: 1,
	}
}

func (this *FreqStack) Push(val int) {
	this.freqMap[val]++
	this.maxFreq = max(this.maxFreq, this.freqMap[val])
	this.bucket[this.freqMap[val]] = append(this.bucket[this.freqMap[val]], val)
}

func (this *FreqStack) Pop() int {
	current := this.bucket[this.maxFreq]
	val := current[len(current)-1]
	current = current[:len(current)-1]
	this.bucket[this.maxFreq] = current
	if len(current) == 0 {
		this.maxFreq--
	}
	this.freqMap[val]--
	return val
}
