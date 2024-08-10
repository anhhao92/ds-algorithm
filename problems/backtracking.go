package problems

import (
	"slices"
	"sort"
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

// LC46
func permute(nums []int) [][]int {
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
func permuteUnique(nums []int) [][]int {
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

type TrieNode struct {
	children map[byte]*TrieNode
	isWord   bool
}

func FindWords(board [][]byte, words []string) []string {
	result := []string{}
	row, col := len(board), len(board[0])
	visited := make([][]bool, row)
	root := &TrieNode{children: make(map[byte]*TrieNode)}
	for i := 0; i < row; i++ {
		visited[i] = make([]bool, col)
	}
	for _, w := range words {
		curNode := root
		for i := 0; i < len(w); i++ {
			char := w[i]
			if curNode.children[char] == nil {
				curNode.children[char] = &TrieNode{children: make(map[byte]*TrieNode)}
			}
			curNode = curNode.children[char]
		}
		curNode.isWord = true
	}
	var dfs func(r, c int, trie *TrieNode, word []byte)
	dfs = func(r, c int, trie *TrieNode, word []byte) {
		if r < 0 || c < 0 || r >= row || c >= col || visited[r][c] {
			return
		}
		visited[r][c] = true
		char := board[r][c]
		curTrie := trie.children[char]
		if curTrie == nil {
			visited[r][c] = false
			return
		}
		word = append(word, char)
		if curTrie.isWord {
			result = append(result, string(word))
			curTrie.isWord = false
		}
		directions := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
		for _, d := range directions {
			dr, dc := r+d[0], c+d[1]
			dfs(dr, dc, curTrie, word)
		}
		visited[r][c] = false
	}
	// traverse cells
	for i := 0; i < row; i++ {
		for j := 0; j < col; j++ {
			dfs(i, j, root, []byte{})
		}
	}
	return result
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
