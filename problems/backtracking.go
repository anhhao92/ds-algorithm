package problems

import (
	"slices"
	"sort"
	"strconv"
	"strings"
)

type LetterCombinations struct {
	phone  map[byte]string
	digits string
	result []string
	stack  []byte
}

func (l *LetterCombinations) dfs(index int) {
	// return when reaching last branch
	if index == len(l.digits) {
		l.result = append(l.result, string(l.stack))
		return
	}
	currentStr := l.phone[l.digits[index]]
	for i := 0; i < len(currentStr); i++ {
		l.stack = append(l.stack, currentStr[i])
		l.dfs(index + 1)
		// pop element from stack then backtrack
		l.stack = l.stack[:len(l.stack)-1]
	}
}

func NewLetterCombinations(digits string) []string {
	l := LetterCombinations{
		digits: digits,
		phone: map[byte]string{
			'2': "abc",
			'3': "def",
			'4': "ghi",
			'5': "jkl",
			'6': "mno",
			'7': "pqrs",
			'8': "tuv",
			'9': "wxyz",
		}}
	if len(l.digits) > 0 {
		l.dfs(0)
	}
	return l.result
}

// Leetcode 22
func NewGenerateParenthesis(n int) []string {
	result := []string{}
	temp := make([]byte, 0, n)
	var backtrack func(open, close int)
	backtrack = func(open, close int) {
		if open == n && close == n {
			result = append(result, string(temp))
			return
		}
		if close < open {
			temp = append(temp, ')')
			backtrack(open, close+1)
			temp = temp[:len(temp)-1]
		}
		if open < n {
			temp = append(temp, '(')
			backtrack(open+1, close)
			temp = temp[:len(temp)-1]
		}
	}
	backtrack(0, 0)
	return result
}

// LC 77
func Combination(n int, k int) [][]int {
	res := [][]int{}
	var backtrack func(combination []int, index int)
	backtrack = func(combination []int, index int) {
		if len(combination) == k {
			res = append(res, slices.Clone(combination)) // have to clone it
			return
		}
		for i := index; i <= n; i++ {
			combination = append(combination, i)
			backtrack(combination, i+1)
			combination = combination[:len(combination)-1]
		}
	}
	backtrack([]int{}, 1)
	return res
}

// LC46
func Permuatation(nums []int) [][]int {
	result := [][]int{}
	var backtrack func(perms []int, index int)
	backtrack = func(perms []int, index int) {
		if index == len(nums) {
			result = append(result, slices.Clone(nums))
		}
		for i := index; i < len(nums); i++ {
			perms[index], perms[i] = perms[i], perms[index] // swap current with the element at position i
			backtrack(perms, index+1)
			perms[index], perms[i] = perms[i], perms[index] // backtracking
		}
	}
	backtrack(nums, 0)
	return result
}

// LC47
func PermuteUnique(nums []int) [][]int {
	result := [][]int{}
	var backtrack func(perms []int, index int)
	backtrack = func(perms []int, index int) {
		if index == len(nums) {
			result = append(result, slices.Clone(perms))
			return
		}
		hash := map[int]bool{}
		for i := index; i < len(nums); i++ {
			if hash[perms[i]] {
				continue
			}
			hash[perms[i]] = true
			perms[index], perms[i] = perms[i], perms[index]
			backtrack(perms, index+1)
			perms[index], perms[i] = perms[i], perms[index]
		}
	}
	backtrack(nums, 0)
	return result
}

// LC 241
func DiffWaysToCompute(expression string) []int {
	res := []int{}
	for i := 0; i < len(expression); i++ {
		if expression[i] == '+' || expression[i] == '-' || expression[i] == '*' {
			n1 := DiffWaysToCompute(expression[:i])
			n2 := DiffWaysToCompute(expression[i+1:])
			for _, v1 := range n1 {
				for _, v2 := range n2 {
					switch expression[i] {
					case '+':
						res = append(res, v1+v2)
					case '-':
						res = append(res, v1-v2)
					case '*':
						res = append(res, v1*v2)
					}
				}
			}
		}
	}
	if len(res) == 0 {
		n, _ := strconv.Atoi(expression)
		res = append(res, n)
	}
	return res
}

// LC 386
func lexicalOrder(n int) []int {
	res := []int{}
	var dfs func(num int)
	dfs = func(num int) {
		if num > n {
			return
		}
		res = append(res, num)
		num = num * 10
		for i := 0; i < 10; i++ {
			dfs(i + num)
		}
	}
	for i := 1; i < 10; i++ {
		dfs(i)
	}
	return res
}

func SolveNQueens(n int) [][]string {
	board := make([][]byte, n)
	result := [][]string{}
	col := make(map[int]bool)
	posDiag := make(map[int]bool) // row + col
	negDiag := make(map[int]bool) // row - col
	for i := range board {
		board[i] = make([]byte, n)
		for j := range board[i] {
			board[i][j] = '.'
		}
	}
	dfsQueenAtRow(0, board, &result, col, posDiag, negDiag)
	return result
}

func dfsQueenAtRow(r int, board [][]byte, result *[][]string, col, posDiag, negDiag map[int]bool) {
	if r == len(board) {
		res := make([]string, len(board))
		for i, v := range board {
			res[i] = string(v)
		}
		*result = append(*result, res)
		return
	}

	for c := range board {
		if col[c] || posDiag[r+c] || negDiag[r-c] {
			continue
		}
		col[c] = true
		posDiag[r+c] = true
		negDiag[r-c] = true
		board[r][c] = 'Q'
		dfsQueenAtRow(r+1, board, result, col, posDiag, negDiag)
		col[c] = false
		posDiag[r+c] = false
		negDiag[r-c] = false
		board[r][c] = '.'
	}
}

func FindItinerary(tickets [][]string) []string {
	adj := make(map[string][]string)
	ans := []string{}

	// Fill the adjacency list
	for i := 0; i < len(tickets); i++ {
		adj[tickets[i][0]] = append(adj[tickets[i][0]], tickets[i][1])
	}

	// Sort the destinations in lexical order
	for key := range adj {
		sort.Strings(adj[key])
	}

	// Use a stack to store the itinerary
	stack := []string{"JFK"}

	for len(stack) > 0 {
		src := stack[len(stack)-1]
		if len(adj[src]) == 0 {
			ans = append(ans, src)
			stack = stack[:len(stack)-1]
		} else {
			dst := adj[src][0]
			adj[src] = adj[src][1:]
			stack = append(stack, dst)
		}
	}
	// Reverse the answer to get the correct order
	slices.Reverse(ans)
	return ans
}

func SolveSudoku(board [][]byte) {
	dfsSudoku(0, 0, board)
}

func dfsSudoku(row int, col int, board [][]byte) bool {
	if col == 9 {
		row++
		col = 0
	}
	if row == 9 {
		return true
	}
	// search next col
	if board[row][col] != '.' {
		return dfsSudoku(row, col+1, board)
	}
	for i := byte('1'); i <= '9'; i++ {
		if isValid(row, col, board, i) {
			board[row][col] = i
			if dfsSudoku(row, col+1, board) {
				return true
			}
			// backtracking
			board[row][col] = '.'
		}
	}
	return false
}

func isValid(row, col int, board [][]byte, current byte) bool {
	for i := 0; i < 9; i++ {
		if board[row][i] == current {
			return false
		}
		if board[i][col] == current {
			return false
		}
	}
	rowStart, rowEnd := findBoxCoor(row)
	colStart, colEnd := findBoxCoor(col)
	for i := rowStart; i < rowEnd; i++ {
		for j := colStart; j < colEnd; j++ {
			if board[i][j] == current {
				return false
			}
		}
	}
	return true
}

func findBoxCoor(coor int) (start, end int) {
	if coor < 3 {
		return 0, 3
	} else if coor < 6 {
		return 3, 6
	} else {
		return 6, 9
	}
}

type PartitionPalindrome struct {
	s   string
	ans [][]string
}

func NewPartitionPalindrome(s string) [][]string {
	p := PartitionPalindrome{s: s, ans: [][]string{}}
	p.dfsPartition(0, []string{})
	return p.ans
}

func (p *PartitionPalindrome) dfsPartition(index int, rs []string) {
	if index == len(p.s) {
		p.ans = append(p.ans, slices.Clone(rs))
		return
	}
	for i := index + 1; i <= len(p.s); i++ {
		subStr := p.s[index:i]
		if p.isPalindrome(subStr) {
			rs = append(rs, subStr)
			p.dfsPartition(i, rs)
			rs = rs[:len(rs)-1]
		}

	}
}

func (p *PartitionPalindrome) isPalindrome(s string) bool {
	n := len(s)
	for i := 0; i < n/2; i++ {
		if s[i] != s[n-i-1] {
			return false
		}
	}
	return true
}

func exist(board [][]byte, word string) bool {
	visited := make([][]bool, len(board))
	for i := range board {
		visited[i] = make([]bool, len(board[0]))
	}
	for i, row := range board {
		for j, c := range row {
			if c == word[0] {
				if dfsWordSearch(i, j, 0, board, visited, word) {
					return true
				}
			}
		}
	}
	return false
}

func dfsWordSearch(r, c, index int, board [][]byte, visited [][]bool, word string) bool {
	row, col := len(board), len(board[0])
	if index == len(word) {
		return true
	}
	if r < 0 || r >= row || c < 0 || c >= col {
		return false
	}
	if visited[r][c] {
		return false
	}
	if board[r][c] != word[index] {
		return false
	}
	visited[r][c] = true
	top := dfsWordSearch(r-1, c, index+1, board, visited, word)
	bottom := dfsWordSearch(r+1, c, index+1, board, visited, word)
	left := dfsWordSearch(r, c-1, index+1, board, visited, word)
	right := dfsWordSearch(r, c+1, index+1, board, visited, word)

	if top || bottom || left || right {
		return true
	}
	visited[r][c] = false
	return false
}

func combinationSum2(candidates []int, target int) [][]int {
	result := [][]int{}
	slices.Sort(candidates)
	var dfs func(cur []int, target int, index int)
	dfs = func(cur []int, target, index int) {
		if target == 0 {
			result = append(result, slices.Clone(cur))
			return
		}
		if target < 0 {
			return
		}
		for i := index; i < len(candidates); i++ {
			if i > index && candidates[i] == candidates[i-1] {
				continue
			}
			cur = append(cur, candidates[i])
			dfs(cur, target-candidates[i], i+1) // i+1 - we don't want to reuse current index
			cur = cur[:len(cur)-1]
		}
	}
	dfs([]int{}, target, 0)
	return result
}

func combinationSum3(k int, n int) [][]int {

	result := [][]int{}
	var dfs func(n, k int, cur []int, sum, num int)
	dfs = func(n, k int, cur []int, sum, num int) {
		if sum == n && len(cur) == k {
			result = append(result, slices.Clone(cur))
			return
		}
		if sum > n {
			return
		}
		for currentNum := num; currentNum <= 9; currentNum++ {
			cur = append(cur, currentNum)
			dfs(n, k, cur, currentNum+sum, currentNum+1)
			cur = cur[:len(cur)-1]
		}
	}
	dfs(n, k, []int{}, 0, 1)
	return result
}

func subsets(nums []int) [][]int {
	result := [][]int{}
	var dfs func(cur []int, index int)
	dfs = func(cur []int, index int) {
		result = append(result, slices.Clone(cur))
		for i := index; i < len(nums); i++ {
			cur = append(cur, nums[i])
			dfs(cur, i+1)
			cur = cur[:len(cur)-1]
		}
	}
	dfs([]int{}, 0)
	return result
}

func subsetsWithDup(nums []int) [][]int {
	result := [][]int{}
	slices.Sort(nums)
	var dfs func(cur []int, index int)
	dfs = func(cur []int, index int) {
		result = append(result, slices.Clone(cur))
		for i := index; i < len(nums); i++ {
			if index < i && nums[i] == nums[i-1] {
				continue
			}
			cur = append(cur, nums[i])
			dfs(cur, i+1)
			cur = cur[:len(cur)-1]
		}
	}
	dfs([]int{}, 0)
	return result
}

// LC 93
func RestoreIpAddresses(s string) []string {
	res := []string{}
	isValid := func(str string) bool {
		if len(str) > 1 && strings.HasPrefix(str, "0") {
			return false
		}
		num, _ := strconv.Atoi(str)
		return num >= 0 && num <= 255
	}
	var backtrack func(index int, addr []string)
	backtrack = func(index int, addr []string) {
		if index == len(s) && len(addr) == 4 {
			res = append(res, strings.Join(addr, "."))
			return
		}
		if len(addr) > 4 {
			return
		}
		for i := index + 1; i <= min(index+3, len(s)); i++ {
			subStr := s[index:i]
			if isValid(subStr) {
				addr = append(addr, subStr)
				backtrack(i, addr)
				addr = addr[:len(addr)-1]
			}
		}
	}
	backtrack(0, []string{})
	return res
}

// LC 1849
func splitStringDescendingConsecutiveValues(s string) bool {
	var backtrack func(index, prev int) bool
	backtrack = func(index, prev int) bool {
		if index == len(s) {
			return true
		}
		for i := index + 1; i <= len(s); i++ {
			subStr := s[index:i]
			val, _ := strconv.Atoi(subStr)
			if val+1 == prev && backtrack(i, val) {
				return true
			}
		}
		return false
	}
	for i := 0; i < len(s)-1; i++ {
		val := s[:i+1]
		prev, _ := strconv.Atoi(val)
		if backtrack(i+1, prev) {
			return true
		}
	}
	return false
}

// LC 1980
func findDifferentBinaryString(nums []string) string {
	hashSet := map[string]bool{}
	for _, v := range nums {
		hashSet[v] = true
	}
	s := make([]byte, len(nums))
	for i := 0; i < len(nums); i++ {
		s[i] = '0'
	}
	var backtrack func(index int) string
	backtrack = func(index int) string {
		if index == len(nums) {
			res := string(s)
			if !hashSet[res] {
				return res
			}
			return ""
		}
		// try 0
		res := backtrack(index + 1)
		if res != "" {
			return res
		}
		// try 1
		s[index] = '1'
		return backtrack(index + 1)
	}
	return backtrack(0)
}

// LC 473
func MakeSquare(matchsticks []int) bool {
	length := 0
	for _, v := range matchsticks {
		length += v
	}
	if length%4 != 0 {
		return false
	}
	length /= 4
	slices.SortFunc(matchsticks, func(a, b int) int {
		return b - a
	})
	sticks := make([]int, 4)
	var backtrack func(index int) bool
	backtrack = func(index int) bool {
		if index == len(matchsticks) {
			return true
		}
		for i := 0; i < 4; i++ {
			// avoid duplicated works
			if matchsticks[index]+sticks[i] > length || i > 0 && sticks[i] == sticks[i-1] {
				continue
			}
			if sticks[i]+matchsticks[index] <= length {
				sticks[i] += matchsticks[index]
				if backtrack(index + 1) {
					return true
				}
				sticks[i] -= matchsticks[index]
			}
			// 2nd optimization
			if sticks[i] == 0 {
				return false
			}
		}
		return false
	}
	return backtrack(0)
}

// LC 698
func CanPartitionKSubsets(nums []int, k int) bool {
	sum := 0
	for _, v := range nums {
		sum += v
	}
	if sum%k != 0 {
		return false
	}
	sum /= k
	subSet := make([]int, k)
	// reverse order
	slices.SortFunc(nums, func(a, b int) int {
		return b - a
	})
	var backtrack func(index int) bool
	backtrack = func(index int) bool {
		if index == len(nums) {
			return true
		}
		for i := 0; i < k; i++ {
			// 1st optimization
			// we traversed subSet[i-1] not the answer so skip identical subSet[i]
			if nums[index]+subSet[i] > sum || i > 0 && subSet[i] == subSet[i-1] {
				continue
			}
			if subSet[i]+nums[index] <= sum {
				subSet[i] += nums[index]
				if backtrack(index + 1) {
					return true
				}
				subSet[i] -= nums[index]
			}
			// 2nd optimization
			if subSet[i] == 0 {
				return false
			}
		}
		return false
	}
	return backtrack(0)
}

// LC 1255
func maxScoreWords(words []string, letters []byte, score []int) int {
	var countLetter [26]int
	var res = 0
	for _, c := range letters {
		countLetter[c-'a']++
	}
	canFormWord := func(usedLetters [26]int) bool {
		for i := range usedLetters {
			if usedLetters[i] > countLetter[i] {
				return false
			}
		}
		return true
	}
	var backtrack func(index int, usedLetters [26]int)
	backtrack = func(index int, usedLetters [26]int) {
		if index == len(words) {
			s := 0
			for i, v := range usedLetters {
				s += v * score[i]
			}
			res = max(res, s)
			return
		}
		// skip
		backtrack(index+1, usedLetters)
		// take
		word := words[index]
		for i := 0; i < len(word); i++ {
			usedLetters[word[i]-'a']++
		}
		if canFormWord(usedLetters) {
			backtrack(index+1, usedLetters)
		}
		for i := 0; i < len(word); i++ {
			usedLetters[word[i]-'a']--
		}
	}
	backtrack(0, [26]int{})
	return res
}
