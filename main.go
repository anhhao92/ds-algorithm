package main

import (
	"fmt"
	"leetcode/problems"
)

func main() {
	a := problems.FindSubstring("lingmindraboofooowingdingbarrwingmonkeypoundcake", []string{"fooo", "barr", "wing", "ding", "wing"})
	fmt.Println("=>", a)
}
