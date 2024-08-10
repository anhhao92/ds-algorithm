package main

import (
	"fmt"
	"leetcode/problems"
)

func main() {
	a := problems.ThreeSumClosest([]int{4, 0, 5, -5, 3, 3, 0, -4, -5}, -2)
	fmt.Println("=>", a)
}
