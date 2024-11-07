package main

import (
	"fmt"
	"leetcode/problems"
)

func main() {
	t1 := problems.MaximumSafenessFactor([][]int{{0, 1, 1}, {0, 1, 1}, {1, 1, 1}})
	t2 := problems.WiggleSortII([]int{1, 5, 1, 1, 6, 4})
	fmt.Println("=>", t1, t2)
}
