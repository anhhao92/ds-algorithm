package problems

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
