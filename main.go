package main

import (
	"fmt"
	"leetcode/problems"
)

func main() {
	a := problems.ShortestBridge([][]int{{0, 1}, {1, 0}})
	fmt.Println("=>", a)
}
