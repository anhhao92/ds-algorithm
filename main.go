package main

import (
	"fmt"
	"leetcode/problems"
)

func main() {
	a := problems.GroupAnagrams([]string{"eat", "tea", "tan", "ate", "nat", "bat"})
	fmt.Println("=>", a)
}
