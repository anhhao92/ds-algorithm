package main

import (
	"fmt"
	"leetcode/problems"
)

func main() {
	a := problems.JobScheduling([]int{1, 2, 3, 3}, []int{3, 4, 5, 6}, []int{50, 10, 40, 70})
	fmt.Println("=>", a)
}
