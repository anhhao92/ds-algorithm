package problems

import (
	"slices"
	"strconv"
)

func WordBreak(s string, wordDict []string) bool {
	dict := map[string]bool{}
	cache := map[string]bool{}
	for _, v := range wordDict {
		dict[v] = true
	}
	return dfs(s, dict, cache)
}

func dfs(s string, dict map[string]bool, cache map[string]bool) bool {
	value, isKeyExist := cache[s]
	if isKeyExist {
		return value
	}
	if len(s) == 0 {
		return true
	}
	for i := 1; i <= len(s); i++ {
		left := s[0:i]
		right := s[i:]
		if dict[left] && dfs(right, dict, cache) {
			cache[s] = true
			return true
		}
	}
	cache[s] = false
	return false
}

func FindAllConcatenatedWordsInADict(words []string) []string {
	dict := map[string]bool{}
	result := []string{}
	for _, v := range words {
		dict[v] = true
	}

	for _, v := range words {
		delete(dict, v)
		cache := map[string]bool{}
		if isConcat(v, dict, cache) {
			result = append(result, v)
		}
		dict[v] = true
	}
	return result
}

func isConcat(word string, dict map[string]bool, cache map[string]bool) bool {
	if value, isCacheKeyExist := cache[word]; isCacheKeyExist {
		return value
	}
	if len(word) == 0 {
		return true
	}
	for i := 1; i <= len(word); i++ {
		left := word[0:i]
		right := word[i:]
		if dict[left] && isConcat(right, dict, cache) {
			cache[word] = true
			return true
		}
	}
	cache[word] = false
	return cache[word]
}

type LongestCommonSubsequence struct {
	text1 string
	text2 string
	dp    [][]int
}

func NewLongestCommonSubsequence(text1 string, text2 string) string {
	l := LongestCommonSubsequence{text1: text1, text2: text2}
	l.longestCommonSubsequence()
	return l.reconstructSubsequence()
}

func (l *LongestCommonSubsequence) longestCommonSubsequence() int {
	text1, text2 := l.text1, l.text2
	l1, l2 := len(text1), len(text2)
	dp := make([][]int, l1+1)
	for i := range dp {
		dp[i] = make([]int, l2+1)
	}
	for i := 1; i <= l1; i++ {
		for j := 1; j <= l2; j++ {
			if text1[i-1] == text2[j-1] {
				dp[i][j] = 1 + dp[i-1][j-1] // 1 + text1[i -1] && text2[j - 1]
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1]) // check text1[i -1] || text2[j - 1]
			}
		}
	}
	l.dp = dp
	return dp[l1][l2]
}

func (l *LongestCommonSubsequence) reconstructSubsequence() string {
	text1, text2, dp := l.text1, l.text2, l.dp
	text := []byte{}
	for i, j := len(text1), len(text2); i > 0 && j > 0; {
		if text1[i-1] == text2[j-1] {
			text = append(text, text1[i-1])
			i--
			j--
		} else if dp[i-1][j] > dp[i][j-1] {
			i--
		} else {
			j--
		}
	}
	slices.Reverse(text)
	return string(text)
}

func numDistinct(s string, t string) int {
	m, n := len(s), len(t)
	dp := make([][]int, n+1)
	for i := 0; i <= n; i++ {
		dp[i] = make([]int, m+1)
	}
	// set 1st row
	for i := range dp[0] {
		dp[0][i] = 1
	}

	for i := 1; i <= n; i++ {
		dp[i][0] = 0
		for j := 1; j <= m; j++ {
			if t[i-1] == s[j-1] {
				dp[i][j] = dp[i-1][j-1] + dp[i][j-1]
			} else {
				dp[i][j] = dp[i][j-1]
			}
		}
	}
	return dp[n][m]
}

func canPartition(nums []int) bool {
	sum := 0

	for i := range nums {
		sum += nums[i]
	}
	if sum%2 != 0 {
		return false
	}
	half := sum / 2
	dp := make([][]bool, len(nums))
	for i := 0; i < len(nums); i++ {
		dp[i] = make([]bool, half+1)
	}
	// set 1st col
	for i := range dp {
		dp[i][0] = true
	}
	// set 1st row
	for i := 1; i < len(dp[0]); i++ {
		if nums[0] == i {
			dp[0][i] = true
		} else {
			dp[0][i] = false
		}
	}

	for i := 1; i < len(dp); i++ {
		for sum = 1; sum < len(dp[0]); sum++ {
			take := false
			diff := sum - nums[i]
			if diff >= 0 {
				take = dp[i-1][diff]
			}
			skip := dp[i-1][sum]
			dp[i][sum] = take || skip
		}
	}
	return dp[len(nums)-1][half]
}

// Unbounded Knapsack
func coinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	for i := 1; i <= amount; i++ {
		dp[i] = amount + 1
	}
	for currentAmount := 1; currentAmount <= amount; currentAmount++ {
		// update min amount for each coin
		for _, coin := range coins {
			if currentAmount-coin >= 0 {
				dp[currentAmount] = min(dp[currentAmount], dp[currentAmount-coin]+1)
			}
		}
	}
	// check if there's a combination to made up the amount
	if dp[amount] < amount+1 {
		return dp[amount]
	}
	return -1
}

// Unbounded Knapsack
func change(coins []int, amount int) int {
	dp := make([]int, amount+1)
	dp[0] = 1
	for i := range coins {
		for currentAmount := 1; currentAmount <= amount; currentAmount++ {
			if currentAmount-coins[i] >= 0 {
				dp[currentAmount] += dp[currentAmount-coins[i]]
			}
		}
	}

	return dp[amount]
}

// Leetcode 497
func findTargetSumWays(nums []int, target int) int {
	// dp := make(map[string]int)
	// return dfsFindTargetSum(0, 0, target, nums, dp)
	// Iterative approach: based on 2 * sum(P) = target + sum(nums) -> sum(P) must be even
	// Find a subset non-continuos P of nums such that sum(P) = (target + sum(nums)) / 2
	sum := 0
	for _, v := range nums {
		sum += v
	}
	if sum < target || (sum+target)%2 != 0 {
		return 0
	}
	target = (sum + target) / 2
	if target < 0 { // sum(P) >= 0
		return 0
	}
	// 0/1 Knapsack
	dp := make([]int, target+1)
	dp[0] = 1
	for _, v := range nums {
		for i := target; i >= v; i-- {
			dp[i] += dp[i-v]
		}
	}
	return dp[target]
}

// leetcode 120
func minimumTotal(triangle [][]int) int {
	height := len(triangle)
	for i := height - 2; i >= 0; i-- {
		for j := 0; j < len(triangle[i]); j++ {
			triangle[i][j] = triangle[i][j] + min(triangle[i+1][j], triangle[i+1][j+1])
		}
	}
	return triangle[0][0]
}

// leetcode 72
func minDistance(word1 string, word2 string) int {
	m, n := len(word1)+1, len(word2)+1
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
	}
	// set 1st row
	for i := 0; i < n; i++ {
		dp[0][i] = i
	}
	// fill the rest table
	for i := 1; i < m; i++ {
		dp[i][0] = i
		for j := 1; j < n; j++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1] // delete current and inherit value
			} else {
				dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1 // min(insert, replace, delete) + 1
			}
		}
	}
	return dp[m-1][n-1]
}

func maxProduct(nums []int) int {
	maxProduct, minProduct, result := nums[0], nums[0], nums[0]
	for i := 1; i < len(nums); i++ {
		localMin := minProduct * nums[i]
		localMax := maxProduct * nums[i]

		minProduct = min(nums[i], localMin, localMax)
		maxProduct = max(nums[i], localMax, localMin)
		result = max(result, maxProduct)
	}
	return result
}

func numSquares(n int) int {
	dp := make([]int, n+1)
	for num := 1; num <= n; num++ {
		for i := 1; i*i <= n; i++ {
			dp[num] = min(dp[num], dp[num-(i*i)]+1)
		}
	}
	return dp[n]
}

func maximalSquare(matrix [][]byte) int {
	m, n := len(matrix), len(matrix[0])
	dp := make([][]int, m+1)
	maxWidth := 0

	for i := 0; i < m+1; i++ {
		dp[i] = make([]int, n+1)
	}

	for i := 1; i < m+1; i++ {
		for j := 1; j < n+1; j++ {
			if matrix[i-1][j-1] == '1' {
				topLeft := dp[i-1][j-1]
				left := dp[i][j-1]
				top := dp[i-1][j]
				dp[i][j] = min(topLeft, top, left) + 1
				maxWidth = max(dp[i][j], maxWidth)
			}
		}
	}
	return maxWidth * maxWidth
}

func countPalindromeSubstrings(s string) int {
	count := 0
	for i := 0; i < len(s); i++ {

		for l, r := i, i; l >= 0 && r < len(s) && s[l] == s[r]; {
			count++
			l--
			r++
		}
		for l, r := i, i+1; l >= 0 && r < len(s) && s[l] == s[r]; {
			count++
			l--
			r++
		}
	}
	return count
}

func rob(nums []int) int {
	rob1, rob2 := 0, 0
	for i := 0; i < len(nums); i++ {
		newRob := max(nums[i]+rob1, rob2)
		rob1 = rob2
		rob2 = newRob
	}
	return rob2
}

func rob2(nums []int) int {
	return max(nums[0], rob(nums[1:]), rob(nums[:len(nums)-1]))
}

func longestIncreasingPath(matrix [][]int) int {
	dp := make([][]int, len(matrix))
	maxPath := 0
	for i := range dp {
		dp[i] = make([]int, len(dp[i]))
	}
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[i]); j++ {
			dfsLongestIncreasingPath(i, j, -1, matrix, dp)
			maxPath = max(maxPath, dp[i][j])
		}
	}
	return maxPath
}

func dfsLongestIncreasingPath(row, col, value int, matrix, dp [][]int) int {
	if row >= len(matrix) || row < 0 || col >= len(matrix[0]) || col < 0 {
		return 0
	}
	if matrix[row][col] <= value {
		return 0
	}

	if dp[row][col] != 0 {
		return dp[row][col]
	}
	top := dfsLongestIncreasingPath(row-1, col, matrix[row][col], matrix, dp)
	bottom := dfsLongestIncreasingPath(row+1, col, matrix[row][col], matrix, dp)
	right := dfsLongestIncreasingPath(row, col+1, matrix[row][col], matrix, dp)
	left := dfsLongestIncreasingPath(row, col-1, matrix[row][col], matrix, dp)
	dp[row][col] = 1 + max(top, right, left, bottom)
	return dp[row][col]
}

func uniquePaths(m int, n int) int {
	row := make([]int, n)
	currentRow := make([]int, n)
	for i := range row {
		row[i] = 1
	}
	for i := m - 2; i >= 0; i-- {
		currentRow[n-1] = 1
		for j := n - 2; j >= 0; j-- {
			currentRow[j] = currentRow[j+1] + row[j]
		}
		row = slices.Clone(currentRow)
	}
	return row[0]
}

func uniquePathsWithObstacles(obstacles [][]int) int {
	m, n := len(obstacles), len(obstacles[0])
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}

	for i := m - 1; i >= 0; i-- {
		for j := n - 1; j >= 0; j-- {
			if obstacles[i][j] == 1 {
				continue
			}
			if i == m-1 && j == n-1 {
				dp[i][j] = 1
			} else {
				dp[i][j] = dp[i+1][j] + dp[i][j+1] // bottom + right
			}
		}
	}
	return dp[0][0]
}

/*
Leetcode 97
s1=ab s2=ae s3=abae
s1->0 1 2
s2  a b
0 a T T T
1 e F F T
2   F F T
*/
func isInterleave(s1 string, s2 string, s3 string) bool {
	if len(s1)+len(s2) != len(s3) {
		return false
	}
	dp := make([][]bool, len(s1)+1)
	for i := range dp {
		dp[i] = make([]bool, len(s2)+1)
	}
	dp[len(s1)][len(s2)] = true
	for i := len(s1); i >= 0; i-- {
		for j := len(s2); j >= 0; j-- {
			if i < len(s1) && s3[i+j] == s1[i] && dp[i+1][j] {
				dp[i][j] = true
			}
			if j < len(s2) && s3[i+j] == s2[j] && dp[i][j+1] {
				dp[i][j] = true
			}
		}
	}

	return dp[0][0]
}

func numDecodings(s string) int {
	n := len(s)
	dp := make([]int, n)
	for i := n - 1; i >= 0; i-- {
		if s[i] == '0' {
			continue
		}
		if i == n-1 {
			dp[i] = 1
			continue
		}
		str := string(s[i : i+2])
		num, _ := strconv.Atoi(str)
		if num > 26 {
			dp[i] = dp[i+1]
		} else if i == n-2 {
			dp[i] = dp[i+1] + 1
		} else {
			dp[i] = dp[i+1] + dp[i+2]
		}
	}
	return dp[0]
}

func maxProfit(prices []int) int {
	n := len(prices)
	maxProfitBuyAt := make([]int, n)
	for buy := n - 2; buy >= 0; buy-- {
		for sell := buy + 1; sell < n; sell++ {
			if prices[buy] > prices[sell] {
				continue
			}
			profit := prices[sell] - prices[buy]
			// cooldown
			if sell+2 < n {
				profit += maxProfitBuyAt[sell+2]
			}
			maxProfitBuyAt[buy] = max(maxProfitBuyAt[buy], profit)
		}
		maxProfitBuyAt[buy] = max(maxProfitBuyAt[buy], maxProfitBuyAt[buy+1])
	}
	return maxProfitBuyAt[0]
}
