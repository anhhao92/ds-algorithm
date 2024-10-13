package main

import (
	"fmt"
	"leetcode/problems"
)

func main() {
	a := problems.FindTargetSumWays([]int{1, 2, 4, 2}, 3)
	fmt.Println("=>", a)
}
