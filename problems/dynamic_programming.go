package problems

import (
	"math"
	"math/bits"
	"slices"
	"strconv"
	"strings"
)

const MOD = int(1e9 + 7)

// 139
func WordBreak(s string, wordDict []string) bool {
	length := len(s)
	dp := make([]bool, length+1)
	dp[length] = true
	for i := length - 1; i >= 0; i-- {
		for _, w := range wordDict {
			if i+len(w) <= length && dp[i+len(w)] && s[i:i+len(w)] == w {
				dp[i] = true
				break
			}
		}
	}
	return dp[0]
}

// 140
func WordBreakII(s string, wordDict []string) []string {
	n := len(s)
	dict := map[string]bool{}
	dp := make([][]string, n+1)
	dp[0] = []string{""}

	for _, v := range wordDict {
		dict[v] = true
	}
	for i := 1; i <= n; i++ {
		list := []string{}
		for j := 0; j < i; j++ {
			subStr := s[j:i]
			if dict[subStr] {
				// start j to i contains word in dict
				for _, w := range dp[j] {
					list = append(list, strings.TrimSpace(w+" "+subStr))
				}
			}
		}
		dp[i] = list
	}
	return dp[n]
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

func LongestCommonSubsequence(text1 string, text2 string) int {
	m, n := len(text1), len(text2)
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if text1[i-1] == text2[j-1] {
				dp[i][j] = 1 + dp[i-1][j-1] // 1 + text1[i -1] && text2[j - 1]
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1]) // check text1[i -1] & text2[j] or text1[i] & text2[j - 1]
			}
		}
	}
	return dp[m][n]
}

// LC 1035
func maxUncrossedLines(nums1 []int, nums2 []int) int {
	m, n := len(nums1), len(nums2)
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if nums1[i-1] == nums2[j-1] {
				dp[i][j] = 1 + dp[i-1][j-1]
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
		}
	}
	return dp[m][n]
}

func ReconstructLongestCommonSubsequence(text1 string, text2 string, dp [][]int) string {
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

// LC516
func LongestPalindromeSubseq(s string) int {
	sb := []byte(s)
	slices.Reverse(sb)
	t := string(sb)
	n := len(s)
	// LCS with 2 slices
	current := make([]int, n+1)
	dp := make([]int, n+1)
	for i := 1; i <= n; i++ {
		for j := 1; j <= n; j++ {
			if s[i-1] == t[j-1] {
				current[j] = 1 + dp[j-1]
			} else {
				current[j] = max(current[j-1], dp[j])
			}
		}
		dp, current = current, dp
	}
	return dp[n]
}

// LC 5
func LongestPalindromeSubstring(s string) string {
	maxCount, start := 0, 0
	for i := 0; i < len(s); i++ {
		for l, r := i, i; l >= 0 && r < len(s) && s[l] == s[r]; {
			if r-l+1 > maxCount {
				maxCount = r - l + 1
				start = l
			}
			l--
			r++
		}
		for l, r := i, i+1; l >= 0 && r < len(s) && s[l] == s[r]; {
			if r-l+1 > maxCount {
				maxCount = r - l + 1
				start = l
			}
			l--
			r++
		}
	}
	return s[start : start+maxCount]
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

/*
LC 322 Unbounded Knapsack
amount = [5 4 3 2 1 0] coin=[1, 3, 5]

	init dp = maxInt
	dp = [1 2 1 2 1 0]
*/
func CoinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	for i := 1; i <= amount; i++ {
		dp[i] = amount + 1
	}
	for currentAmount := 1; currentAmount <= amount; currentAmount++ {
		for _, coin := range coins {
			// update min amount for each coin
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

/*
	LC 518 Unbounded Knapsack

amount x 5 4 3 2 1|0
coin  |1|3 2 2 1 1|1 (use coin 1 or skip)
coin  |3|1 0 1 0 0|1
coin  |5|1 0 0 0 0|1
*/
func CoinChangeII(coins []int, amount int) int {
	dp := make([]int, amount+1)
	dp[0] = 1 // basecase there's one way to sum up 0 amount
	// optimized 1D array because we only need 1 row below from range 1 -> amount
	for i := range coins {
		for currentAmount := 1; currentAmount <= amount; currentAmount++ {
			if currentAmount-coins[i] >= 0 {
				dp[currentAmount] += dp[currentAmount-coins[i]]
			}
		}
	}

	return dp[amount]
}

// Leetcode 494
/* [1,2,4,2] sum = 6
sum	0 1 2 3 4 5 6
  1|1|1 0 0 0 0 0
  2|1|1 1 1 0 0 0
  4|1|1 1 1 1 1 1
  2|1|1 2 2 2 2 2 (sum=6 pick(6-2) + skip 6)
*/
func FindTargetSumWays(nums []int, target int) int {
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
	// must filled from right to left
	for _, v := range nums {
		for i := target; i >= v; i-- {
			dp[i] += dp[i-v]
		}
	}
	return dp[target]
}

// LC474
func findMaxForm(strs []string, m int, n int) int {
	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
	}
	for _, s := range strs {
		mCount, nCount := strings.Count(s, "0"), strings.Count(s, "1")
		for i := m; i >= mCount; i-- {
			for j := n; j >= nCount; j-- {
				dp[i][j] = max(1+dp[i-mCount][j-nCount], dp[i][j])
			}
		}
	}
	return dp[m][n]
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

// LC279
func numSquares(n int) int {
	dp := make([]int, n+1)
	for num := 1; num <= n; num++ {
		current := n
		for i := 1; i*i <= num; i++ {
			current = min(current, 1+dp[num-(i*i)])
		}
		dp[num] = current
	}
	return dp[n]
}

// LC377
func combinationSum4(nums []int, target int) int {
	dp := make([]int, target+1)
	dp[0] = 1
	for i := 1; i <= target; i++ {
		for _, num := range nums {
			if i-num >= 0 {
				dp[i] += dp[i-num]
			}
		}
	}
	return dp[target]
}

// LC 221
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

// LC740
func deleteAndEarn(nums []int) int {
	count := map[int]int{}
	for _, v := range nums {
		count[v]++
	}
	slices.Sort(nums)
	e1, e2 := 0, 0
	for i := 0; i < len(nums); i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		earn := nums[i] * count[nums[i]]
		temp := e2
		if i > 0 && nums[i-1]+1 == nums[i] {
			e2 = max(earn+e1, e2)
			e1 = temp
		} else {
			e2 = earn + e2
			e1 = temp
		}
	}
	return e2
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

// LC309
/*
[]noStock: Can buy or rest
[]hasStock: Can sell or cooldown
[]justSold: Only cooldown
*/
func BestTimeToBuyStockWithCooldown(prices []int) int {
	n := len(prices)
	noStock, hasStock, justSold := make([]int, n), make([]int, n), make([]int, n)
	hasStock[0] = -prices[0] // buy at index 0 so profit -prices[0]
	for i := 1; i < n; i++ {
		noStock[i] = max(noStock[i-1], justSold[i-1])
		hasStock[i] = max(hasStock[i-1], noStock[i-1]-prices[i])
		justSold[i] = hasStock[i-1] + prices[i]
	}
	return max(noStock[n-1], justSold[n-1])
}

// LC122
func BestTimeToBuyStockII(prices []int) int {
	profit := 0
	for i := 1; i < len(prices); i++ {
		if prices[i] > prices[i-1] {
			profit += prices[i] - prices[i-1]
		}
	}
	return profit
}

// LC 188/123 O(n*k)
func BestTimeToBuyStockIV(k int, prices []int) int {
	n := len(prices)
	dp := make([][]int, k+1)
	for i := 0; i <= k; i++ {
		dp[i] = make([]int, n)
	}
	for i := 1; i <= k; i++ {
		maxDiff := -prices[0]
		for j := 1; j < n; j++ {
			dp[i][j] = max(dp[i][j-1], maxDiff+prices[j])
			maxDiff = max(maxDiff, dp[i-1][j]-prices[j])
			// profit := 0
			// for m := 0; m < j; m++ {
			// 	profit = max(profit, prices[j]-prices[m]+dp[i-1][m])
			// }
			// // O(n*k*k) not making tx or complete tx by (buying at m and sell at j) + previos tx at day m
			// dp[i][j] = max(dp[i][j-1], profit)
		}
	}
	return dp[k][n-1]
}

// LC 714
// noStock  <-> hasStock
func BestTimeToBuyStockWithFee(prices []int, fee int) int {
	n := len(prices)
	noStock, hasStock := make([]int, n), make([]int, n)
	hasStock[0] = -prices[0] // buy at index 0 so profit -prices[0]
	for i := 1; i < n; i++ {
		noStock[i] = max(noStock[i-1], hasStock[i-1]+prices[i]-fee)
		hasStock[i] = max(hasStock[i-1], noStock[i]-prices[i])
	}
	return noStock[n-1]
}

/*
LC1049
  0 1 2 3 4 5 6 7 8 9 10 11
2 0 0 2 2 2 2 2 2 2 2 2  2
7 0 0 2 2 2 2 2 7 7 9 9  9
*/

func LastStoneWeightII(stones []int) int {
	sum := 0
	for _, n := range stones {
		sum += n
	}
	half := sum / 2
	n := len(stones)
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, half+1)
	}
	for i := 1; i <= n; i++ {
		for s := 1; s <= half; s++ {
			if stones[i-1] > s {
				dp[i][s] = dp[i-1][s]
			} else {
				dp[i][s] = max(dp[i-1][s], dp[i-1][s-stones[i-1]]+stones[i-1]) // skip or take
			}
		}
	}
	return sum - 2*dp[n][half]
}

// LC983
func mincostTickets(days []int, costs []int) int {
	n := len(days)
	dp := make([]int, n+1)
	cover := []int{1, 7, 30}
	for i := n - 1; i >= 0; i-- {
		dp[i] = math.MaxInt32
		for j, c := range costs {
			k := i
			for k < n && days[k] < days[i]+cover[j] {
				k++
			}
			dp[i] = min(dp[i], c+dp[k])
		}
	}

	return dp[0]
	// var dfs func(index int) int
	// dfs = func(index int) int {
	// 	if index == -1 {
	// 		return 0
	// 	}
	// 	if dp[index] != 0 {
	// 		return dp[index]
	// 	}
	// 	dp[index] = math.MaxInt32
	// 	for i, v := range costs {
	// 		nextIndex := slices.IndexFunc(days, func(e int) bool {
	// 			return e >= days[index]+cover[i]
	// 		})
	// 		dp[index] = min(dp[index], v+dfs(nextIndex))
	// 	}
	// 	return dp[index]
	// }
}

// LC877
func stoneGame(piles []int) bool {
	total := 0
	n := len(piles)
	dp := make([][]int, n)
	for i := range piles {
		dp[i] = make([]int, n)
		total += piles[i]
	}
	var dfs func(left, right int) int
	dfs = func(left, right int) int {
		if left == right {
			return 0
		}
		if dp[left][right] != 0 {
			return dp[left][right]
		}
		takeFirst, takeLast := 0, 0
		if (right-left+1)%2 == 0 {
			takeFirst = piles[left]
			takeLast = piles[right]
		}
		dp[left][right] = max(takeFirst+dfs(left+1, right), takeLast+dfs(left, right-1))
		return dp[left][right]
	}
	return dfs(0, n-1) > total/2
}

// LC 1140
func stoneGameII(piles []int) int {
	n := len(piles)
	suffixSum := make([]int, n+1)
	for i := n - 1; i >= 0; i-- {
		suffixSum[i] = suffixSum[i+1] + piles[i]
	}
	dp := make([][]int, n)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	for i := n - 1; i >= 0; i-- {
		for m := 1; m <= n; m++ {
			if i+2*m >= n {
				dp[i][m] = suffixSum[i]
			} else {
				for x := 1; x <= 2*m; x++ {
					dp[i][m] = max(dp[i][m], suffixSum[i]-dp[i+x][max(m, x)])
				}
			}
		}
	}
	return dp[0][1]
}

// LC 1406
func stoneGameIII(stoneValue []int) string {
	n := len(stoneValue)
	dp := make([]int, n+1) //dp[i] = Alice - Bob
	for i := n - 1; i >= 0; i-- {
		dp[i] = stoneValue[i] - dp[i+1]
		if i+2 <= n {
			dp[i] = max(dp[i], stoneValue[i]+stoneValue[i+1]-dp[i+2])
		}
		if i+3 <= n {
			dp[i] = max(dp[i], stoneValue[i]+stoneValue[i+1]+stoneValue[i+2]-dp[i+3])
		}
	}
	if dp[0] > 0 {
		return "Alice"
	} else if dp[0] < 0 {
		return "Bob"
	}
	return "Tie"
}

// LC64
func minPathSum(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	for i := m - 1; i >= 0; i-- {
		for j := n - 1; j >= 0; j-- {
			if i == m-1 && j == n-1 {
				continue
			}
			if i+1 >= m {
				grid[i][j] += grid[i][j+1]
			} else if j+1 >= n {
				grid[i][j] += grid[i+1][j]
			} else {
				grid[i][j] += min(grid[i+1][j], grid[i][j+1])
			}
		}
	}
	return grid[0][0]
}

// LC1911
func maxAlternatingSum(nums []int) int64 {
	// dp := make(map[[2]int]int)
	// var dfs func(index, sign int) int
	// dfs = func(index, sign int) int {
	// 	if index >= len(nums) {
	// 		return 0
	// 	}
	// 	key := [2]int{index, sign}
	// 	if v, ok := dp[key]; ok {
	// 		return v
	// 	}
	// 	take := sign*nums[index] + dfs(index+1, -1*sign)
	// 	skip := dfs(index+1, sign)
	// 	dp[key] = max(take, skip)
	// 	return dp[key]
	// }
	// return int64(dfs(0, 1))
	sumEven, sumOdd := 0, 0
	for _, num := range nums {
		tmpEven := max(sumOdd+num, sumEven) // take even or skip
		tmpOdd := max(sumEven-num, sumOdd)
		sumEven, sumOdd = tmpEven, tmpOdd
	}
	return int64(sumEven)
}

// LC1155
func NumRollsToTarget(n int, k int, target int) int {
	mod := int(1e9 + 7)
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, target+1)
	}
	dp[0][0] = 1
	for i := 1; i <= n; i++ {
		for j := 1; j <= target; j++ {
			count := 0
			for t := 1; t <= k; t++ {
				if j-t >= 0 {
					count = (count + dp[i-1][j-t]) % mod
				}
			}
			dp[i][j] = count
		}
	}
	return dp[n][target]
}

// LC 343
func integerBreak(n int) int {
	dp := make([]int, n+1)
	dp[1] = 1
	for num := 2; num <= n; num++ {
		dp[num] = 1
		for i := 1; i < num; i++ { // should not break n
			dp[num] = max(dp[num], max(dp[i], i)*(num-i))
		}
	}
	return dp[n]
}

// LC576
func findPaths(m int, n int, maxMove int, startRow int, startColumn int) int {
	mod := int(1e9 + 7)
	dp, temp := make([][]int, m), make([][]int, m)
	for i := range dp {
		dp[i] = make([]int, n)
		temp[i] = make([]int, n)
	}
	dp[startRow][startColumn] = 1
	ans := 0
	for move := 1; move <= maxMove; move++ {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				if i == m-1 {
					ans = (ans + dp[i][j]) % mod
				}
				if j == n-1 {
					ans = (ans + dp[i][j]) % mod
				}
				if i == 0 {
					ans = (ans + dp[i][j]) % mod
				}
				if j == 0 {
					ans = (ans + dp[i][j]) % mod
				}
				up, down, left, right := 0, 0, 0, 0
				if i > 0 {
					up = dp[i-1][j]
				}
				if j > 0 {
					left = dp[i][j-1]
				}
				if i < m-1 {
					right = dp[i+1][j]
				}
				if j < n-1 {
					down = dp[i][j+1]
				}
				temp[i][j] = (up + down + left + right) % mod
			}
		}
		dp, temp = temp, dp
	}
	return ans
}

// LC2370
func longestIdealString(s string, k int) int {
	dp := [26]int{}
	res := 0
	for i := 0; i < len(s); i++ {
		current := int(s[i] - 'a')
		longest := 0
		for prev := 0; prev < len(dp); prev++ {
			if int(math.Abs(float64(current-prev))) <= k {
				longest = max(longest, 1+dp[prev])
			}
		}
		dp[current] = max(longest, dp[current])
		res = max(res, dp[current])

	}
	return res
}

// LC 300
func LongestIncreasingSubSequence(nums []int) int {
	n := len(nums)
	dp := make([]int, n)
	res := 0
	for i := n - 1; i >= 0; i-- {
		dp[i] = 1
		for j := i + 1; j < n; j++ {
			if nums[i] < nums[j] {
				dp[i] = max(dp[i], 1+dp[j])
			}
		}
		res = max(res, dp[i])
	}
	return res
}

// LC 673
func CountNumberOfLIS(nums []int) int {
	n := len(nums)
	dp := make([][2]int, n)
	maxLIS, maxCount := 0, 0
	for i := n - 1; i >= 0; i-- {
		maxCurrentLIS, maxCurrentCount := 1, 1
		for j := i + 1; j < n; j++ {
			if nums[i] < nums[j] {
				length, cnt := dp[j][0], dp[j][1]
				if length+1 > maxCurrentLIS {
					maxCurrentLIS = length + 1
					maxCurrentCount = cnt
				} else if length+1 == maxCurrentLIS {
					maxCurrentCount += cnt
				}
			}
		}
		dp[i] = [2]int{maxCurrentLIS, maxCurrentCount}
		if maxLIS < maxCurrentLIS {
			maxLIS = maxCurrentLIS
			maxCount = maxCurrentCount
		} else if maxLIS == maxCurrentLIS {
			maxCount += maxCurrentCount
		}
	}
	return maxCount
}

// LC1220
func countVowelPermutation(n int) int {
	a, e, i, o, u := 1, 1, 1, 1, 1
	for j := 1; j < n; j++ {
		a_next := e
		e_next := (a + i) % MOD
		i_next := (a + e + o + u) % MOD
		o_next := (i + u) % MOD
		u_next := a
		a, e, i, o, u = a_next, e_next, i_next, o_next, u_next
	}
	return (a + e + i + o + u) % MOD
}

// LC1866
func rearrangeSticks(n int, k int) int {
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, k+1)
	}
	dp[0][0] = 1
	for i := 1; i <= n; i++ {
		for j := 1; j <= k; j++ {
			dp[i][j] = (dp[i-1][j-1] + (i-1)*dp[i-1][j]) % MOD
		}
	}
	return dp[n][k] % MOD
}

// LC 1856
func maxSumMinProduct(nums []int) int {
	n := len(nums)
	prefix := make([]int, n+1)
	for i, v := range nums {
		prefix[i+1] = prefix[i] + v
	}
	stack := [][2]int{} // [index, val] increasing order stack
	res := 0
	for i := 0; i < n; i++ {
		newStart := i
		for len(stack) > 0 && stack[len(stack)-1][1] > nums[i] {
			start, val := stack[len(stack)-1][0], stack[len(stack)-1][1]
			sum := prefix[i] - prefix[start]
			res = max(res, sum*val)
			newStart = start
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, [2]int{newStart, nums[i]})
	}

	for _, v := range stack {
		start, val := v[0], v[1]
		sum := prefix[n] - prefix[start]
		res = max(res, sum*val)
	}
	return res % MOD
}

// LC 2707
func MinExtraChar(s string, dictionary []string) int {
	n := len(s)
	dict := map[string]bool{}
	dp := make([]int, n+1)
	for _, v := range dictionary {
		dict[v] = true
	}
	for i := 1; i <= n; i++ {
		// skip current char
		res := 1 + dp[i-1]
		for j := 0; j < i; j++ {
			if dict[s[j:i]] {
				res = min(res, dp[j])
			}
		}
		dp[i] = res
	}
	return dp[n]
}

// LC 2369
func validPartition(nums []int) bool {
	n := len(nums)
	dp := make([]bool, n+1)
	dp[n] = true
	for i := n - 2; i >= 0; i-- {
		if nums[i] == nums[i+1] {
			dp[i] = dp[i] || dp[i+2]
			if i+2 < len(nums) && nums[i+1] == nums[i+2] {
				dp[i] = dp[i] || dp[i+3]
			}
		}
		if i+2 < n && nums[i]+1 == nums[i+1] && nums[i+1]+1 == nums[i+2] {
			dp[i] = dp[i] || dp[i+3]
		}
	}
	return dp[0]
}

// LC 691
func minStickers(stickers []string, target string) int {
	const maxSticker = int(1e9 + 7)
	n := 1 << len(target)
	dp := make([]int, n)
	// for i := range dp {
	// 	dp[i] = -1
	// }
	// dp[0] = 0
	// for mask := range n {
	// 	if dp[mask] == -1 {
	// 		continue
	// 	}
	// 	for _, s := range stickers {
	// 		var stickerSet [26]int
	// 		for i := 0; i < len(s); i++ {
	// 			stickerSet[s[i]-'a']++
	// 		}
	// 		currentMask := mask
	// 		for i := 0; i < len(target); i++ {
	// 			// current character wasn't set
	// 			c := target[i] - 'a'
	// 			if mask&(1<<i) == 0 && stickerSet[c] != 0 {
	// 				stickerSet[c]--
	// 				currentMask ^= 1 << i
	// 			}
	// 		}
	// 		// more bits were set
	// 		if dp[currentMask] == -1 || dp[currentMask] > dp[mask]+1 {
	// 			dp[currentMask] = dp[mask] + 1
	// 		}
	// 	}
	// }
	// return dp[n-1]
	var dfs func(mask int) int
	dfs = func(mask int) int {
		// done if all bits set to 1
		if mask == n-1 {
			return 0
		}
		if dp[mask] != 0 {
			return dp[mask]
		}
		// in case we don't find any match sticker
		dp[mask] = maxSticker
		for _, s := range stickers {
			var stickerSet [26]int
			for i := 0; i < len(s); i++ {
				stickerSet[s[i]-'a']++
			}
			currentMask := mask
			for i := 0; i < len(target); i++ {
				// current character wasn't set
				c := target[i] - 'a'
				if mask&(1<<i) == 0 && stickerSet[c] != 0 {
					stickerSet[c]--
					currentMask ^= 1 << i
				}
			}
			// more bits were set
			if currentMask > mask {
				dp[mask] = min(dp[mask], 1+dfs(currentMask))
			}
		}
		return dp[mask]
	}
	res := dfs(0)
	if res != maxSticker {
		return res
	}
	return -1
}

// LC 2140
func mostPoints(questions [][]int) int64 {
	n := len(questions)
	dp := make([]int, n+1)
	for i := n - 1; i >= 0; i-- {
		point, skip := questions[i][0], questions[i][1]+1
		if i+skip < n {
			dp[i] = max(dp[i+1], dp[i+skip]+point)
		} else {
			dp[i] = max(dp[i+1], point)
		}
	}
	return int64(dp[0])
}

// LC 2466
func countGoodStrings(low int, high int, zero int, one int) int {
	dp := make([]int, high+1)
	dp[0] = 1
	for i := 1; i <= high; i++ {
		// zero=1, one=2 -> xx0 or xx11
		if i >= zero {
			dp[i] += dp[i-zero]
		}
		if i >= one {
			dp[i] += dp[i-one]
		}
	}
	res := 0
	for i := low; i <= high; i++ {
		res += dp[i]
	}
	return res % MOD
}

// LC 1626
func bestTeamScore(scores []int, ages []int) int {
	n := len(scores)
	scoreAge := make([][2]int, n)
	for i := range scoreAge {
		scoreAge[i] = [2]int{scores[i], ages[i]}
	}
	slices.SortFunc(scoreAge, func(a, b [2]int) int {
		// scores are equal then sort by age
		if a[0] == b[0] {
			return a[1] - b[1]
		}
		return a[0] - b[0]
	})
	dp := make([]int, n)
	for i := range dp {
		dp[i] = scoreAge[i][0]
	}
	for i := 0; i < n; i++ {
		maxScore, maxAge := scoreAge[i][0], scoreAge[i][1]
		for j := 0; j < i; j++ {
			age := scoreAge[j][1]
			if age <= maxAge {
				dp[i] = max(dp[i], maxScore+dp[j])
			}
		}
	}
	return slices.Max(dp)
}

// LC 1048
func LongestStrChain(words []string) int {
	slices.SortFunc(words, func(a, b string) int {
		return len(a) - len(b)
	})
	dp := map[string]int{}
	res := 1
	for _, w := range words {
		dp[w] = 1
		for i := range w {
			pred := w[:i] + w[i+1:]
			if val, ok := dp[pred]; ok {
				dp[w] = max(dp[w], val+1)
			}
		}
		res = max(res, dp[w])
	}
	return res
}

// LC 935
func knightDialer(n int) int {
	jumps := [...][]int{
		{4, 6},
		{6, 8},
		{7, 9},
		{4, 8},
		{3, 9, 0},
		{},
		{1, 7, 0},
		{2, 6},
		{1, 3},
		{2, 4},
	}
	const MOD = int(1e9 + 7)
	var dp [10]int
	for i := range dp {
		dp[i] = 1
	}

	for range n - 1 {
		var next [10]int
		for i, jump := range jumps {
			for _, cell := range jump {
				next[i] = (next[i] + dp[cell]) % MOD
			}
		}
		dp = next
	}
	sum := 0
	for _, v := range dp {
		sum = (sum + v) % MOD
	}
	return sum % MOD
}

// LC 688
func knightProbability(n int, k int, row int, column int) float64 {
	moves := [][2]int{{-2, 1}, {-2, -1}, {-1, 2}, {-1, -2}, {2, 1}, {2, -1}, {1, 2}, {1, -2}}
	next := make([][]float64, n)
	dp := make([][]float64, n)
	for i := range n {
		dp[i] = make([]float64, n)
		next[i] = make([]float64, n)
	}
	dp[row][column] = 1
	for range k {
		for i := range n {
			for j := range n {
				next[i][j] = 0
				for _, move := range moves {
					// check prev move
					r, c := i-move[0], j-move[1]
					if r >= 0 && r < n && c >= 0 && c < n {
						next[i][j] += dp[r][c] / 8.0
					}
				}
			}
		}
		dp, next = next, dp
	}
	poss := 0.0
	for i := range n {
		for j := range n {
			poss += dp[i][j]
		}
	}
	return poss
}

// LC 368
func largestDivisibleSubset(nums []int) []int {
	slices.Sort(nums)
	n := len(nums)
	dp := make([][]int, n)
	res := []int{}
	for i := n - 1; i >= 0; i-- {
		dp[i] = []int{nums[i]}
		for j := i + 1; j < n; j++ {
			if nums[j]%nums[i] == 0 {
				if len(dp[j])+1 > len(dp[i]) {
					dp[i] = append([]int{nums[i]}, dp[j]...)
				}
			}
		}
		if len(dp[i]) > len(res) {
			res = dp[i]
		}
	}
	return res
}

// LC 1799 bitmask
func maxScore(nums []int) int {
	n := len(nums)
	dp := make([]int, 1<<n)
	var gcd = func(x, y int) int {
		for y != 0 {
			x, y = y, x%y
		}
		return x
	}
	for mask := 1; mask < len(dp); mask++ {
		c := bits.OnesCount(uint(mask))
		if c%2 != 0 {
			continue
		}
		step := c / 2 // (110011) -> step = 2
		for i := 0; i < n-1; i++ {
			if mask&(1<<i) == 0 {
				continue
			}
			for j := i + 1; j < n; j++ {
				if mask&(1<<j) == 0 {
					continue
				}
				prevMask := mask // (110011) -> prev 000011
				prevMask ^= 1 << i
				prevMask ^= 1 << j
				dp[mask] = max(dp[mask], step*gcd(nums[i], nums[j])+dp[prevMask])
			}
		}
	}
	return dp[(1<<n)-1]
	// var dfs func(mask, step int) int
	// dfs = func(mask, step int) int {
	// 	if dp[mask] != 0 {
	// 		return dp[mask]
	// 	}
	// 	var res int
	// 	for i := 0; i < n-1; i++ {
	// 		if mask&(1<<i) != 0 {
	// 			continue
	// 		}
	// 		for j := i + 1; j < n; j++ {
	// 			if mask&(1<<j) != 0 {
	// 				continue
	// 			}
	// 			nextMask := mask | 1<<i | 1<<j
	// 			res = max(res, step*gcd(nums[i], nums[j])+dfs(nextMask, step+1))
	// 			dp[mask] = res
	// 		}
	// 	}
	// 	return res
	// }
}

// LC 1964
func longestObstacleCourseAtEachPosition(obstacles []int) []int {
	dp := []int{}
	res := []int{}
	// LIS variant
	for _, obstacle := range obstacles {
		index := BinarySearch(dp, obstacle+1) // upper bound
		if index == len(dp) {
			dp = append(dp, obstacle)
		} else {
			dp[index] = obstacle
		}
		res = append(res, index+1)
	}
	return res
}

// LC 1359
func countOrders(n int) int {
	slots := 2 * n
	res := 1
	// Put a pair (P1-P2) into s slot will be s*(s-1)/2 choices
	for slots > 0 {
		validChoices := slots * (slots - 1) / 2
		res *= validChoices
		slots -= 2
	}
	return res % MOD
}

// LC 2147
func numberOfWays(corridor string) int {
	dp := []int{}
	res := 1
	for i, c := range corridor {
		if c == 'S' {
			dp = append(dp, i)
		}
	}
	if len(dp) < 2 || len(dp)%2 == 1 {
		return 0
	}
	for i := 1; i+1 < len(dp); i += 2 {
		res *= dp[i+1] - dp[i]
		res = res % MOD
	}
	return res % MOD
}

// LC 1235
func JobScheduling(startTime []int, endTime []int, profit []int) int {
	n := len(startTime)
	intervals := make([][3]int, n)
	dp := make([]int, n+1)
	for i := range n {
		intervals[i][0] = startTime[i]
		intervals[i][1] = endTime[i]
		intervals[i][2] = profit[i]
	}
	slices.SortFunc(intervals, func(a, b [3]int) int {
		return a[0] - b[0]
	})
	for i := range intervals {
		startTime[i] = intervals[i][0]
	}
	for i := len(startTime) - 1; i >= 0; i-- {
		dp[i] = intervals[i][2]
		endTime := intervals[i][1]
		index, _ := slices.BinarySearch(startTime, endTime)
		if index < len(startTime) {
			dp[i] += dp[index]
		}
		dp[i] = max(dp[i], dp[i+1])
	}
	return dp[0]
}

// LCC 552
/* L
A |1|1|0
  |1|0|0
*/
func checkRecord(n int) int {
	var dp = [2][3]int{{1, 1, 0}, {1, 0, 0}}
	var next [2][3]int
	for i := 0; i < n-1; i++ {
		// pick P we must reset L when picking P
		next[0][0] = (dp[0][0] + dp[0][1] + dp[0][2]) % MOD
		next[1][0] = (dp[1][0] + dp[1][1] + dp[1][2]) % MOD
		// pick L
		next[0][1] = dp[0][0]
		next[0][2] = dp[0][1]
		next[1][1] = dp[1][0]
		next[1][2] = dp[1][1]
		// pick A
		next[1][0] += (dp[0][0] + dp[0][1] + dp[0][2]) % MOD
		dp, next = next, dp
	}
	res := 0
	for _, row := range dp {
		for _, v := range row {
			res = (res + v) % MOD
		}
	}
	return res
}

// LC 926
func minFlipsMonoIncreasing(s string) int {
	res, countOne := 0, 0
	for i := 0; i < len(s); i++ {
		// if we get 1 it doesn't matter what came before
		// ex 00[1], 11[1]
		if s[i] == '1' {
			countOne++
		} else {
			res = min(countOne, res+1) // flip all 1 or flip SoFar + 1
		}
	}
	return res
}

// LC 2218
func maxValueOfCoins(piles [][]int, k int) int {
	n := len(piles)
	dp := make([][]int, n)
	for i := range dp {
		dp[i] = make([]int, k+1)
	}
	var dfs func(index, coin int) int
	dfs = func(index, coin int) int {
		if index == len(piles) {
			return 0
		}
		if dp[index][coin] != 0 {
			return dp[index][coin]
		}
		dp[index][coin] = dfs(index+1, coin)
		sum := 0
		for i := 0; i < min(coin, len(piles[index])); i++ {
			sum += piles[index][i]
			dp[index][coin] = max(dp[index][coin], sum+dfs(index+1, coin-i-1))
		}
		return dp[index][coin]

	}
	return dfs(0, k)
}

// LC 10
func isMatchRegex(s string, p string) bool {
	dp := make(map[[2]int]bool)
	var dfs func(i, j int) bool
	dfs = func(i, j int) bool {
		if i >= len(s) && j >= len(p) {
			return true
		}
		if j >= len(p) {
			return false
		}
		key := [2]int{i, j}
		if v, ok := dp[key]; ok {
			return v
		}
		dp[key] = false
		isMatch := i < len(s) && (s[i] == p[j] || p[j] == '.')
		if j+1 < len(p) && p[j+1] == '*' {
			dp[key] = dfs(i, j+2) || (isMatch && dfs(i+1, j))
			return dp[key]
		}
		if isMatch {
			dp[key] = dfs(i+1, j+1)
			return dp[key]
		}
		return dp[key]
	}
	return dfs(0, 0)
}

// LC 312
func maxCoinsBrustBallons(nums []int) int {
	n := len(nums)
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	var dfs func(l, r int) int
	dfs = func(l, r int) int {
		if l > r {
			return 0
		}
		if dp[l][r] != 0 {
			return dp[l][r]
		}
		for i := l; i <= r; i++ {
			left, right := 1, 1
			// as we pop i last so left=l-1, right=r+1
			if l-1 >= 0 {
				left = nums[l-1]
			}
			if r+1 < n {
				right = nums[r+1]
			}
			coin := left*nums[i]*right + dfs(l, i-1) + dfs(i+1, r)
			dp[l][r] = max(dp[l][r], coin)
		}
		return dp[l][r]
	}
	return dfs(0, n-1)
}

// LC1639 https://leetcode.com/problems/number-of-ways-to-form-a-target-string-given-a-dictionary/
func NumWaysToFormTargetString(words []string, target string) int {
	n := len(words[0])
	count := make([][]int, n)
	for i := 0; i < n; i++ {
		count[i] = make([]int, 26)
		for _, word := range words {
			count[i][int(word[i]-'a')]++
		}
	}
	dp := make([]int, len(target)+1)
	dp[len(target)] = 1
	for i := n - 1; i >= 0; i-- {
		for j, c := range target {
			dp[j] += dp[j+1] * count[i][c-'a']
			dp[j] %= MOD
		}
	}
	return dp[0]
}

// LC 1547
/*
0 1 --- 6 n
*/
func minCostCutStick(n int, cuts []int) int {
	cuts = append(cuts, 0, n)
	slices.Sort(cuts)
	n = len(cuts)
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	for diff := 2; diff < n; diff++ {
		for l := 0; l+diff < n; l++ {
			r := l + diff
			dp[l][r] = math.MaxInt32
			for k := l + 1; k < r; k++ {
				dp[l][r] = min(dp[l][r], cuts[r]-cuts[l]+dp[l][k]+dp[k][r])
			}
		}
	}
	return dp[0][n-1]

	// var dfs func(l, r int) int
	// dfs = func(l, r int) int {
	// 	if r-l == 1 {
	// 		return 0
	// 	}
	// 	key := [2]int{l, r}
	// 	if v, ok := dp[key]; ok {
	// 		return v
	// 	}
	// 	res := math.MaxInt32
	// 	for _, c := range cuts {
	// 		if l < c && c < r {
	// 			res = min(res, r-l+dfs(l, c)+dfs(c, r))
	// 		}
	// 	}
	// 	if res == math.MaxInt32 {
	// 		res = 0
	// 	}
	// 	dp[key] = res
	// 	return res
	// }
	// return dfs(0, n)
}

// LC 2742
func PaintWalls(cost []int, time []int) int {
	n := len(cost)
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	// base case i == n
	for i := 1; i <= n; i++ {
		dp[n][i] = math.MaxInt32
	}
	for i := n - 1; i >= 0; i-- {
		for remain := 1; remain <= n; remain++ {
			paint := cost[i] + dp[i+1][max(remain-time[i]-1, 0)]
			skip := dp[i+1][remain]
			dp[i][remain] = min(paint, skip)
		}
	}
	return dp[0][n]
}

// LC 1269
func numWaysToStayTheSamePlace(steps int, arrLen int) int {
	// dfs(i, steps - 1) + dfs(i + 1, steps - 1) + dfs(i - 1, steps - 1)
	arrLen = min(steps, arrLen)
	dp := make([]int, arrLen)
	next := make([]int, arrLen)
	dp[0] = 1
	for step := 1; step <= steps; step++ {
		for i := 0; i < arrLen; i++ {
			next[i] = dp[i]
			if i-1 >= 0 {
				next[i] = (next[i] + dp[i-1]) % MOD
			}
			if i+1 < arrLen {
				next[i] = (next[i] + dp[i+1]) % MOD
			}
		}
		dp, next = next, dp
	}
	return dp[0] % MOD
}

// LC 920
func numMusicPlaylists(n int, goal int, k int) int {
	dp := make([][]int, goal+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
		for j := range dp[i] {
			dp[i][j] = -1
		}
	}

	var dfs func(g, oldSong int) int
	dfs = func(g, oldSong int) int {
		if g == 0 && oldSong == n {
			return 1
		}
		if g == 0 || oldSong > n {
			return 0
		}
		if dp[g][oldSong] != -1 {
			return dp[g][oldSong]
		}
		// choose new
		res := (n - oldSong) * dfs(g-1, oldSong+1)
		// choose old
		if oldSong > k {
			res += (oldSong - k) * dfs(g-1, oldSong)
		}
		dp[g][oldSong] = res % MOD
		return res
	}
	return dfs(goal, 0)
}

// LC 1463
func cherryPickup(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	dp := make([][]int, n)
	next := make([][]int, n)
	for i := range dp {
		dp[i] = make([]int, n)
		next[i] = make([]int, n)
	}
	delta := [...]int{-1, 0, 1}
	for r := m - 1; r >= 0; r-- {
		for c1 := 0; c1 < n-1; c1++ {
			for c2 := c1 + 1; c2 < n; c2++ {
				maxCherries := 0
				cherries := grid[r][c1] + grid[r][c2]
				for _, d1 := range delta {
					for _, d2 := range delta {
						nextCol1, nextCol2 := c1+d1, c2+d2
						if nextCol1 < 0 || nextCol2 == n {
							continue
						}
						maxCherries = max(maxCherries, cherries+dp[nextCol1][nextCol2])
					}
				}
				next[c1][c2] = maxCherries
			}
		}
		dp, next = next, dp
	}
	return dp[0][n-1]
}

// LC 514
func findRotateSteps(ring string, key string) int {
	dp, next := make([]int, len(ring)), make([]int, len(ring))
	var pos [26][]int
	for i := 0; i < len(ring); i++ {
		pos[ring[i]-'a'] = append(pos[ring[i]-'a'], i)
	}
	for k := len(key) - 1; k >= 0; k-- {
		for r := 0; r < len(ring); r++ {
			next[r] = 1_000_000
			for _, i := range pos[key[k]-'a'] {
				dist := min(abs(r-i), len(ring)-abs(r-i))
				next[r] = min(next[r], 1+dist+dp[i])
			}
		}
		dp, next = next, dp
	}
	return dp[0]
	// var dfs func(index, k int) int
	// dfs = func(index, k int) int {
	// 	if k == len(key) {
	// 		return 0
	// 	}
	// 	if dp[index][k] != 0 {
	// 		return dp[index][k]
	// 	}
	// 	res := math.MaxInt32
	// 	for i := 0; i < len(ring); i++ {
	// 		if ring[i] == key[k] {
	// 			dist := min(abs(index-i), len(ring)-abs(index-i))
	// 			res = min(res, 1+dist+dfs(i, k+1))
	// 		}
	// 	}
	// 	dp[index][k] = res
	// 	return res
	// }
	// return dfs(0, 0)
}

// LC 629
func kInversePairs(n int, k int) int {
	dp := make([]int, k+1)
	next := make([]int, k+1)
	dp[0] = 1
	for i := 1; i <= n; i++ {
		total := 0
		for j := 0; j <= k; j++ {
			if j >= i {
				total -= dp[j-i]
			}
			total += dp[j]
			next[j] = total % MOD
		}
		dp, next = next, dp
	}
	return dp[k]
	// var dfs func(num, l int) int
	// dfs = func(num, l int) int {
	// 	if num == 0 {
	// 		if l == 0 {
	// 			return 1
	// 		}
	// 		return 0
	// 	}
	// 	if l < 0 {
	// 		return 0
	// 	}
	// 	res := 0
	// 	for i := 0; i < num; i++ {
	// 		res = (res + dfs(num-1, l-i)) % MOD
	// 	}
	// 	return res
	// }
	// return dfs(n, k)
}
