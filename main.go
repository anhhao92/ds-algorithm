package main

import (
	"fmt"
	"leetcode/problems"
)

func main() {
	a := problems.NumUniqueEmails([]string{"test.email+alex@leetcode.com", "test.e.mail+bob.cathy@leetcode.com", "testemail+david@lee.tcode.com"})
	fmt.Println("=>", a)
}
