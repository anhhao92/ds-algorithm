package main

import (
	"fmt"
	"leetcode/problems"
)

func main() {
	a := problems.FindLadders("hit", "cog", []string{"hot", "dot", "dog", "lot", "log", "cog"})
	fmt.Println("=>", a)
}
