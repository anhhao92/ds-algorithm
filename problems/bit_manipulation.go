package problems

func CountOne(n int) int {
	count := 0
	for n > 0 {
		n = n & (n - 1)
		count++
	}
	return count
}

func FindPositionDistinctKCharacter(s string, k int) int {
	var state = 0
	for i := 0; i < k-1; i++ {
		state ^= 1 << (s[i] % 32)
	}

	for i := 0; i <= len(s)-k; i++ {
		last := s[i+k-1]
		state ^= 1 << (last % 32)
		if CountOne(state) == k {
			return i
		}
		first := s[i]
		state ^= 1 << (first % 32)
	}
	return -1
}

// LC 1239
func MaxLengthConcatenatedUniqString(arr []string) int {
	mask := []int{}
	for _, word := range arr {
		// store distinct char into 26 bit
		state := 0
		for _, c := range word {
			// repeated characters
			if state&(1<<(c-'a')) != 0 {
				state = 0
				break
			}
			state |= 1 << (c - 'a')
		}
		if state != 0 {
			mask = append(mask, state)
		}
	}
	res := 0
	var dfs func(index int, currentConcat int)
	dfs = func(index, currentConcat int) {
		for i := index; i < len(mask); i++ {
			if currentConcat&mask[i] == 0 {
				nextConcat := currentConcat | mask[i]
				res = max(res, CountOne(nextConcat))
				dfs(i+1, nextConcat)
			}
		}
	}
	dfs(0, 0)
	return res
}

// LC 187
func FindRepeatedDnaSequences(s string) []string {
	hash := map[int]int{}
	res := []string{}
	hashKey := 0
	appendCharacter := func(key int, c byte) int {
		key <<= 2  // allocate spaces for 2 bits
		switch c { // append last 2 bits
		case 'C':
			key |= 1
		case 'G':
			key |= 2
		case 'T':
			key |= 3
		}
		key &= 1<<20 - 1 // bits exceed 20 will be 0
		return key
	}
	// sliding window width=9
	for i := 0; i < len(s) && i < 9; i++ {
		hashKey = appendCharacter(hashKey, s[i])
	}

	for i := 9; i < len(s); i++ {
		hashKey = appendCharacter(hashKey, s[i])
		if hash[hashKey] == 1 {
			res = append(res, s[i-9:i+1])
		}
		hash[hashKey]++
	}
	return res
}

// LC 136
func singleNumber(nums []int) int {
	v := 0
	// XOR operator will cancel the numbers appear twice
	for i := 0; i < len(nums); i++ {
		v ^= nums[i]
	}
	return v
}

// LC 260
func singleNumberIII(nums []int) []int {
	xor := 0
	for i := 0; i < len(nums); i++ {
		xor ^= nums[i]
	}
	// xor = 2^3
	diff := 1
	// find the first right most diff bit
	for xor&diff == 0 {
		diff = diff << 1

	}
	a, b := 0, 0
	// duplicated nums must be always in the same group no matter what group it is
	for _, v := range nums {
		if diff&v == diff {
			a = a ^ v
		} else {
			b = b ^ v
		}

	}
	return []int{a, b}
}

// LC 201
func rangeBitwiseAnd(left int, right int) int {
	i := 0
	for left != right {
		left = left >> 1
		right = right >> 1
		i++
	}
	return left << i
}

// LC 190
func reverseBits(num uint32) uint32 {
	res := uint32(0)
	for i := 0; i < 32; i++ {
		bit := (num >> i) & 1
		res |= bit << (31 - i)
	}
	return res
}

// LC 386
func findTheDifference(s string, t string) byte {
	var res byte
	var i int
	for i = 0; i < len(s); i++ {
		res ^= s[i]
		res ^= t[i]
	}
	return res ^ t[i]
}

// LC 338
func countBits(n int) []int {
	dp := make([]int, n+1)
	offset := 1
	for i := 1; i <= n; i++ {
		if offset*2 == i {
			offset = i
		}
		dp[i] = 1 + dp[i-offset]
	}
	return dp
}
