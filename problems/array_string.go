package problems

import (
	"math"
	"slices"
	"strconv"
	"strings"
)

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

// LC 921
func sortArray(nums []int) []int {
	if len(nums) <= 1 {
		return nums
	}
	mid := len(nums) / 2
	left := sortArray(nums[:mid])
	right := sortArray(nums[mid:])
	mergedArray := make([]int, 0, len(nums))
	l, r := 0, 0
	for l < len(left) && r < len(right) {
		if left[l] < right[r] {
			mergedArray = append(mergedArray, left[l])
			l++
		} else {
			mergedArray = append(mergedArray, right[r])
			r++
		}
	}
	for l < len(left) {
		mergedArray = append(mergedArray, left[l])
		l++
	}
	for r < len(right) {
		mergedArray = append(mergedArray, right[r])
		r++
	}
	return mergedArray
}

// LC 1189
func maxNumberOfBalloons(text string) int {
	var b, a, l, o, n int
	for _, c := range text {
		switch c {
		case 'b':
			b++
		case 'a':
			a++
		case 'l':
			l++
		case 'o':
			o++
		case 'n':
			n++
		}
	}
	return min(b, a, l/2, o/2, n)
}

// 290
func wordPattern(pattern string, s string) bool {
	words := strings.Split(s, " ")
	if len(words) != len(pattern) {
		return false
	}
	charToWord := map[byte]string{}
	wordToChar := map[string]byte{}
	for i, w := range words {
		c := pattern[i]
		if word, ok := charToWord[c]; ok {
			if word != w {
				return false
			}
		} else {
			charToWord[c] = w
		}

		if char, ok := wordToChar[w]; ok {
			if char != c {
				return false
			}
		} else {
			wordToChar[w] = c
		}
	}
	return true
}

// LC 539
func FindMinDifference(timePoints []string) int {
	maxMinute, minMinute := 0, 24*60
	timeMinutes := [24 * 60]bool{}
	for _, time := range timePoints {
		arr := strings.Split(time, ":")
		h, _ := strconv.Atoi(arr[0])
		m, _ := strconv.Atoi(arr[1])
		minute := h*60 + m
		maxMinute = max(maxMinute, minute)
		minMinute = min(minMinute, minute)
		if timeMinutes[minute] {
			return 0
		}
		timeMinutes[minute] = true
	}
	prev := maxMinute
	minDiff := (24*60 - maxMinute) + minMinute // distance min <-> max minutes
	for current := maxMinute - 1; current >= minMinute; current-- {
		if timeMinutes[current] {
			minDiff = min(minDiff, prev-current)
			prev = current
		}
	}
	return minDiff
}

func isAnagram(s string, t string) bool {
	hash := map[rune]int{}
	for _, r := range s {
		hash[r]++
	}
	for _, r := range t {
		hash[r]--
		if hash[r] == 0 {
			delete(hash, r)
		}
	}
	return len(hash) == 0
}

// LC724
func findPivotIndex(nums []int) int {
	leftSum, total := 0, 0
	for _, v := range nums {
		total += v
	}
	for i, num := range nums {
		rightSum := total - num - leftSum
		if rightSum == leftSum {
			return i
		}
		leftSum += num
	}
	return -1
}

// LC179
func largestNumber(nums []int) string {
	comparator := func(a int, b int) int {
		stra, strb := strconv.Itoa(a), strconv.Itoa(b)
		if stra+strb > strb+stra {
			return -1
		} else {
			return 1
		}
	}
	slices.SortFunc(nums, comparator)
	var sb strings.Builder
	for _, v := range nums {
		sb.WriteString(strconv.Itoa(v))
	}
	res := sb.String()
	if strings.HasPrefix(res, "0") {
		return "0"
	}
	return res
}

/*
LC 229
Since we have more than n/3 times then we have atmost two elements
Why? Because if we have 3 major elements then 3*(n/3) > array size
We're going to use Boyer-Moore Majority Voting Algorithm
This algorithm can be used to return the highest K elements
that appeared in the array more than array_size/(K+1) times. In our case, K = 2
The major Intuition behind this algorithm is that maintaining voting variable for the candidates:
- Increase the variable if you faced the candidate in your iteration.
- Decrease the variable if you faced another element.
- If the variable reaches 0, look for another promising candidate.
*/
func MajorityElement(nums []int) []int {
	voting1, voting2 := 0, 0
	candidate1, candidate2 := 0, 0
	// first pass is to find potential candidate the majority elements
	for _, value := range nums {
		if candidate1 == value {
			voting1++
		} else if candidate2 == value {
			voting2++
		} else if voting1 == 0 {
			voting1 = 1
			candidate1 = value
		} else if voting2 == 0 {
			voting2 = 1
			candidate2 = value
		} else {
			voting1--
			voting2--
		}
	}
	// second pass is to count the majority elements
	res := []int{}
	voting1, voting2 = 0, 0
	for _, v := range nums {
		if v == candidate1 {
			voting1++
		} else if v == candidate2 {
			voting2++
		}
	}
	if voting1 > len(nums)/3 {
		res = append(res, candidate1)
	}
	if voting2 > len(nums)/3 {
		res = append(res, candidate2)
	}
	return res
}

func LongestEvenSubsequence(nums []int) int {
	minOdd, maxOdd, minEven, maxEven := math.MaxInt32, math.MinInt32, math.MaxInt32, math.MinInt32
	for _, v := range nums {
		if v%2 == 0 {
			minEven = min(minEven, v)
			maxEven = max(maxEven, v)
		} else {
			minOdd = min(minOdd, v)
			maxOdd = max(maxOdd, v)
		}
	}

	odd, even := 0, 0
	for _, v := range nums {
		if minEven <= v && v <= maxEven {
			even++
		}
		if minOdd <= v && v <= maxOdd {
			odd++
		}
	}

	return max(odd, even)
}

func FarestDistanceBetweenZeroOne(nums []int) int {
	j, count := 0, 0
	for i := 1; i < len(nums); i++ {
		if nums[i] == 1 {
			if nums[j] == 1 {
				count = max(count, i-j)
			}
			j = i
		}
	}
	return count
}

func TopKFrequent(nums []int, k int) []int {
	hs := make(map[int]int)
	res := []int{}
	for _, v := range nums {
		hs[v]++
	}
	// bucket sort
	bucket := make(map[int][]int)
	maxFreq := 1
	for key, value := range hs {
		bucket[value] = append(bucket[value], key)
		maxFreq = max(maxFreq, value)
	}
	for i := maxFreq; i > 0 && k > 0; i-- {
		if value, hasKey := bucket[i]; hasKey {
			res = append(res, value...)
			k -= len(value)
		}
	}
	return res
}

func FrequencySort(s string) string {
	res := []byte{}
	hs := make(map[byte]int)
	for i := 0; i < len(s); i++ {
		hs[s[i]]++
	}
	// bucket sort
	bucket := make(map[int][]byte)
	for key, value := range hs {
		bucket[value] = append(bucket[value], key)
	}
	for i := len(s); i > 0; i-- {
		if value, ok := bucket[i]; ok {
			for _, c := range value {
				for j := 0; j < i; j++ {
					res = append(res, c)
				}
			}
		}
	}
	return string(res)
}

// LC238
func ProductExceptSelf(nums []int) []int {
	n := len(nums)
	result := make([]int, n)
	result[0] = 1
	for i := 1; i < n; i++ {
		result[i] = result[i-1] * nums[i-1]
	}
	postfix := 1
	for i := n - 1; i >= 0; i-- {
		result[i] *= postfix
		postfix *= nums[i]
	}
	return result
}

// LC128
func longestConsecutive(nums []int) int {
	setInt := make(map[int]bool)
	maxLen := 0
	for _, v := range nums {
		setInt[v] = true
	}
	for _, v := range nums {
		if !setInt[v-1] {
			length := 0
			for setInt[v+length] {
				length++
			}
			maxLen = max(maxLen, length)
		}
	}
	return maxLen
}

// LC 442
func findDuplicates(nums []int) []int {
	res := []int{}
	for i := 0; i < len(nums); i++ {
		index := abs(nums[i]) - 1
		if nums[index] < 0 {
			res = append(res, abs(nums[i]))
		}
		// mark visited
		nums[index] *= -1
	}
	return res
}

// LC 1249
func minRemoveToMakeValidParenthesis(s string) string {
	stack := []int{}
	result := []byte{}
	// store indices of () into stack so we can remove it after that
	for i := 0; i < len(s); i++ {
		if len(stack) > 0 && s[i] == ')' && s[stack[len(stack)-1]] == '(' {
			stack = stack[:len(stack)-1]
		} else if s[i] == '(' || s[i] == ')' {
			stack = append(stack, i)
		}
	}
	// remove the character equal stack index in the stack
	for i := len(s) - 1; i >= 0; i-- {
		if len(stack) > 0 && stack[len(stack)-1] == i {
			stack = stack[:len(stack)-1]
		} else {
			result = append(result, s[i])
		}
	}
	slices.Reverse(result)
	return string(result)
}

// LC1291
func SequentialDigits(low int, high int) []int {
	res := []int{}
	queue := make([]int, 9)
	for i := range 9 {
		queue[i] = i + 1
	}
	for len(queue) > 0 {
		num := queue[0]
		queue = queue[1:]
		if low <= num && num <= high {
			res = append(res, num)
		}
		lastDigit := num % 10
		if lastDigit < 9 {
			queue = append(queue, num*10+lastDigit+1)
		}
	}
	// seq := "123456789"
	// for i := 0; i < len(seq); i++ {
	// 	for j := i + 1; j < len(seq); j++ {
	// 		n, e := strconv.Atoi(seq[i : j+1])
	// 		if e != nil {
	// 			return res
	// 		}
	// 		if n > high {
	// 			break
	// 		}
	// 		if n >= low {
	// 			res = append(res, n)
	// 		}
	// 	}
	// }
	// slices.Sort(res)
	return res
}

// LC 2306
func NamingCompanyDistinctNames(ideas []string) int64 {
	group := [26]map[string]bool{}
	count := 0
	// init group
	for i := range group {
		group[i] = make(map[string]bool)
	}
	// Group by 1st character [0: {offee, afe}, 1: {aby, est} ]
	for _, idea := range ideas {
		index := idea[0] - 'a'
		suffix := idea[1:]
		group[index][suffix] = true
	}
	for i := 0; i < len(group)-1; i++ {
		for j := i + 1; j < len(group); j++ {
			commonSuffix := 0
			for idea1 := range group[i] {
				if group[j][idea1] {
					commonSuffix++
				}
			}
			distinc1 := len(group[i]) - commonSuffix
			distinc2 := len(group[j]) - commonSuffix
			count += 2 * distinc1 * distinc2
		}
	}
	return int64(count)
}

// LC 2971
func FindLargestPerimeterPolygon(nums []int) int64 {
	slices.Sort(nums)
	maxPerimeter := -1
	curSum := 0
	for i := 0; i < len(nums); i++ {
		if curSum > nums[i] {
			maxPerimeter = curSum + nums[i]
		}
		curSum += nums[i]
	}
	return int64(maxPerimeter)
}

// LC 1685
func getSumAbsoluteDifferences(nums []int) []int {
	n := len(nums)
	prefix := make([]int, n)
	postfix := make([]int, n)
	res := []int{}
	prefix[0] = nums[0]
	for i := 1; i < n; i++ {
		prefix[i] = prefix[i-1] + nums[i]
	}
	postfix[n-1] = nums[n-1]
	for i := n - 2; i >= 0; i-- {
		postfix[i] = postfix[i+1] + nums[i]
	}
	for i := 0; i < n; i++ {
		leftSize, rightSize := i, n-i-1
		res = append(res, leftSize*nums[i]-prefix[i]+postfix[i]-rightSize*nums[i])
	}
	return res
}

// LC41/LC645
func firstMissingPositive(nums []int) int {
	n := len(nums)
	// we're going to use negative and zero in nums to optimize space complexity
	// 1st change all negative to zero
	for i := range nums {
		if nums[i] < 0 {
			nums[i] = 0
		}
	}
	// 2nd change postive value to negative
	for i := range nums {
		index := abs(nums[i]) - 1
		if index >= 0 && index < n {
			if nums[index] > 0 {
				nums[index] *= -1
			} else if nums[index] == 0 { // possible ans is [1..n+1] so we can change it to -1*(n+1)
				nums[index] = -1 * (n + 1)
			}
		}
	}
	// after 2nd pass all nums [1..n] has been marked as negative
	// in case nums[i] >= 0 that is the 1st missing number
	for i := 0; i < n; i++ {
		if nums[i] >= 0 {
			return i + 1
		}
	}
	//all nums are negative so n+1 is the answer
	return n + 1
}

// LC 2348
func zeroFilledSubarray(nums []int) int64 {
	count, curSub := 0, 0
	for _, num := range nums {
		if num == 0 {
			curSub++
		} else {
			curSub = 0
		}
		count += curSub
	}
	return int64(count)
}

// LC 665
func checkPossibility(nums []int) bool {
	count := 0
	for i := 0; i < len(nums)-1; i++ {
		if nums[i] > nums[i+1] {
			if count == 1 {
				return false
			}
			count++
			// 3 [4] 2 4
			if i == 0 || nums[i+1] >= nums[i-1] {
				nums[i] = nums[i+1]
			} else {
				nums[i+1] = nums[i]
			}
		}
	}
	return true
}

// LC1461
func hasAllCodes(s string, k int) bool {
	hs := map[string]bool{}
	for i := 0; i < len(s)-k+1; i++ {
		hs[s[i:i+k]] = true
	}
	return len(hs) == int(math.Pow(2, float64(k)))
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

// LC 438
func FindAnagrams(s string, p string) []int {
	countP, countS := [26]int{}, [26]int{}
	res := []int{}
	if len(s) < len(p) {
		return res
	}
	for i := 0; i < len(p); i++ {
		countP[p[i]-'a']++
		countS[s[i]-'a']++
	}
	for i := len(p); i <= len(s); i++ {
		isValid := true
		for j := 0; j < len(countP); j++ {
			if countP[j] != countS[j] {
				isValid = false
				break
			}
		}
		if isValid {
			res = append(res, i-len(p))
		}
		if i < len(s) {
			countS[s[i]-'a']++
			countS[s[i-len(p)]-'a']--
		}
	}
	return res
}

// LC 2273
func removeAnagrams(words []string) []string {
	res := []string{}
	var lastAnagram [26]int
	for _, word := range words {
		var currentAnagram [26]int
		for i := 0; i < len(word); i++ {
			currentAnagram[word[i]-'a']++
		}
		if currentAnagram != lastAnagram {
			res = append(res, word)
			lastAnagram = currentAnagram
		}
	}
	return res
}

// LC 2405
func OptimalPartitionString(s string) int {
	count := 1
	hash := [26]bool{}
	for i := 0; i < len(s); i++ {
		if hash[s[i]-'a'] {
			count++
			clear(hash[:])
		}
		hash[s[i]-'a'] = true
	}
	return count
}

// LC 2870
func minOperationsToMakeArrayEmpty(nums []int) int {
	count := map[int]int{}
	for _, v := range nums {
		count[v]++
	}
	op := 0
	for _, v := range count {
		if v == 1 {
			return -1
		}
		op += int(math.Ceil(float64(v) / 3.0))
	}
	return op
}

// LC 2966
func divideArray(nums []int, k int) [][]int {
	n := len(nums)
	slices.Sort(nums)
	res := make([][]int, 0, n/3)
	for i := 0; i < n; i += 3 {
		if nums[i+2]-nums[i] > k {
			return [][]int{}
		}
		res = append(res, nums[i:i+3])
	}
	return res
}

// LC 2610
func findMatrix(nums []int) [][]int {
	hash := map[int]int{}
	length := 0
	for _, v := range nums {
		hash[v]++
		length = max(length, hash[v])
	}
	res := make([][]int, length)
	for k, v := range hash {
		for i := 0; i < v; i++ {
			res[i] = append(res[i], k)
		}
	}
	return res
}

// LC 2017
func GridGame(grid [][]int) int64 {
	var res int64 = math.MaxInt64
	col := len(grid[0])
	topSum, bottomSum := int64(0), int64(0)
	for i := 0; i < col; i++ {
		topSum += int64(grid[0][i])
	}
	for i := 0; i < col; i++ {
		topSum -= int64(grid[0][i])
		res = min(res, max(topSum, bottomSum))
		bottomSum += int64(grid[1][i])
	}
	return res
}

// LC 554
func LeastBricks(wall [][]int) int {
	gapCount := map[int]int{}
	maxGap := 0
	for _, bricks := range wall {
		total := 0
		for i := 0; i < len(bricks)-1; i++ {
			total += bricks[i]
			gapCount[total]++
			maxGap = max(maxGap, gapCount[total])
		}
	}
	return len(wall) - maxGap
}

// LC 1963
func minSwaps(s string) int {
	close, maxClose := 0, 0
	for i := 0; i < len(s); i++ {
		if s[i] == ']' {
			close++
		} else {
			close--
		}
		maxClose = max(maxClose, close)

	}
	return (maxClose + 1) / 2
}

// LC 214
func ShortestPalindrome(s string) string {
	isPalindrome := func(s string) bool {
		for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
			if s[i] != s[j] {
				return false
			}
		}
		return true
	}

	for endingIndex := len(s); endingIndex >= 0; endingIndex-- {
		if isPalindrome(s[:endingIndex]) {
			suffix := []byte(s[endingIndex:])
			slices.Reverse(suffix)
			return string(suffix) + s
		}
	}
	return s
}

// LC 1930
func countPalindromicSubsequence(s string) int {
	var left, right [26]byte
	seen := map[[2]byte]bool{}
	for i := 0; i < len(s); i++ {
		right[s[i]-'a']++
	}
	for i := 0; i < len(s)-1; i++ {
		index := s[i] - 'a'
		right[index]--
		for c := 0; c < 26; c++ {
			pair := [2]byte{s[i], byte('a' + c)}
			if left[c] > 0 && right[c] > 0 && !seen[pair] {
				seen[pair] = true
			}
		}
		left[index]++
	}
	return len(seen)
}

// LC 2001
func interchangeableRectangles(rectangles [][]int) int64 {
	count := map[float64]int64{}
	res := int64(0)
	for i := 0; i < len(rectangles); i++ {
		ratio := float64(rectangles[i][0]) / float64(rectangles[i][1])
		res += count[ratio]
		count[ratio]++
	}
	return res
}

// LC 2002 bitmask
func MaxProductOfTwoPalindrome(s string) int {
	n := 1 << len(s)
	palinSubSequences := [][2]int{} // {mask, len}
	isPalindrome := func(s []byte) bool {
		for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
			if s[i] != s[j] {
				return false
			}
		}
		return true
	}
	for mask := 1; mask < n; mask++ {
		subSeq := []byte{}
		for i := 0; i < len(s); i++ {
			// find bit 1 to build the subsequence
			if mask&(1<<i) > 0 {
				subSeq = append(subSeq, s[i])
			}
		}

		if isPalindrome(subSeq) {
			palinSubSequences = append(palinSubSequences, [2]int{mask, len(subSeq)})
		}
	}
	res := 0
	n = len(palinSubSequences)
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			mask1, mask2 := palinSubSequences[i][0], palinSubSequences[j][0]
			if mask1&mask2 == 0 {
				len1, len2 := palinSubSequences[i][1], palinSubSequences[j][1]
				res = max(res, len1*len2)
			}
		}
	}

	return res
}

// LC 799
func champagneTower(poured int, query_row int, query_glass int) float64 {
	prev := []float64{float64(poured)}
	for row := 1; row < query_row+1; row++ {
		cur := make([]float64, row+1)
		for i := range row {
			extra := prev[i] - 1
			if extra > 0 {
				cur[i] += extra / 2
				cur[i+1] += extra / 2
			}
		}
		prev = cur
	}
	return min(1, prev[query_glass])
}

// LC 68
func fullJustify(words []string, maxWidth int) []string {
	res := []string{}
	lines := []string{}
	currentWidth := 0
	for _, word := range words {
		if len(word)+len(lines)+currentWidth > maxWidth {
			totalWord := len(lines)
			space := (maxWidth - currentWidth) / max(1, totalWord-1)
			extraSpace := (maxWidth - currentWidth) % max(1, totalWord-1)
			for i := 0; i < max(1, totalWord-1); i++ {
				lines[i] += strings.Repeat(" ", space)
				if extraSpace > 0 {
					lines[i] += " "
					extraSpace--
				}
			}
			res = append(res, strings.Join(lines, ""))
			lines = []string{}
			currentWidth = 0
		}
		currentWidth += len(word)
		lines = append(lines, word)
	}
	lastLine := strings.Join(lines, " ")
	if len(lastLine) < maxWidth {
		lastLine += strings.Repeat(" ", maxWidth-len(lastLine))
	}
	res = append(res, lastLine)
	return res
}

// LC 8
func myAtoi(s string) int {
	res := 0
	isNegative := false
	var i int = 0
	// skip spaces
	for i < len(s) && (s[i] == ' ') {
		i++
	}
	if i < len(s) {
		if s[i] == '+' {
			i++
		} else if s[i] == '-' {
			isNegative = true
			i++
		}
	}
	maxInt32Prefix, maxInt32Suffix := math.MaxInt32/10, math.MaxInt32%10
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		digit := int(s[i] - '0')
		if isNegative {
			if res > maxInt32Prefix || res == maxInt32Prefix && digit-1 > maxInt32Suffix {
				return math.MinInt32
			}
		} else {
			if res > maxInt32Prefix || res == maxInt32Prefix && digit > maxInt32Suffix {
				return math.MaxInt32
			}
		}
		res = res*10 + digit
		i++
	}
	if isNegative {
		return -res
	}
	return res
}

// LC 3043
func FindLengthLongestCommonPrefix(arr1 []int, arr2 []int) int {
	prefix := map[int]bool{}
	if len(arr1) > len(arr2) {
		arr1, arr2 = arr2, arr1
	}
	for _, v := range arr1 {
		for v > 0 && !prefix[v] {
			prefix[v] = true
			v = v / 10
		}
	}
	length := 0
	for _, v := range arr2 {
		for v > 0 {
			if prefix[v] {
				str := strconv.Itoa(v)
				length = max(length, len(str))
				break
			}
			v = v / 10
		}
	}
	return length
}

func arrayRankTransform(arr []int) []int {
	sortArr := slices.Clone(arr)
	slices.Sort(sortArr)
	sortArr = slices.Compact(sortArr)
	hashMap := map[int]int{}
	for i := 0; i < len(sortArr); i++ {
		hashMap[sortArr[i]] = i + 1
	}
	for i := range arr {
		arr[i] = hashMap[arr[i]]
	}
	return arr
}
