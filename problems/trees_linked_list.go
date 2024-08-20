package problems

import (
	"fmt"
	"strconv"
	"strings"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// LC98
func isValidBST(root *TreeNode) bool {
	prev := 0
	isInitValue := false
	var dfs func(cur *TreeNode) bool
	dfs = func(cur *TreeNode) bool {
		if cur == nil {
			return true
		}
		left := dfs(cur.Left)
		if !left {
			return false
		}
		// inorder traversal
		if !isInitValue {
			prev = cur.Val
			isInitValue = true
		} else if prev >= cur.Val {
			return false
		}
		prev = cur.Val
		right := dfs(cur.Right)
		return right
	}
	return dfs(root)
}

// LC 124
func maxPathSum(root *TreeNode) int {
	maxVal := root.Val
	var dfs func(current *TreeNode) int
	dfs = func(current *TreeNode) int {
		if current == nil {
			return 0
		}
		maxLeft := max(dfs(current.Left), 0)
		maxRight := max(dfs(current.Right), 0)
		maxVal = max(maxVal, maxLeft+maxRight+current.Val)
		return max(maxLeft, maxRight) + current.Val
	}
	dfs(root)
	return maxVal
}

// LC297
// Serializes a tree to a single string.
func serialize(root *TreeNode) string {
	if root == nil {
		return "X"
	}
	left := serialize(root.Left)
	right := serialize(root.Right)
	return fmt.Sprint(root.Val, ",", left, ",", right)
}

// Deserializes your encoded data to tree.
// 1,2,x,x,3,4,5,x,x,x,x
func deserialize(data string) *TreeNode {
	arr := strings.Split(data, ",")
	index := 0
	var dfs func() *TreeNode
	dfs = func() *TreeNode {
		if index >= len(arr) || arr[index] == "X" {
			index++
			return nil
		}
		val, _ := strconv.Atoi(arr[index])
		index++
		node := &TreeNode{Val: val}
		node.Left = dfs()
		node.Right = dfs()
		return node
	}
	return dfs()
}

// LC144 [root -> left -> right]
func PreorderTraversal(root *TreeNode) []int {
	res := []int{}
	if root == nil {
		return res
	}
	stack := []*TreeNode{root}
	for len(stack) > 0 {
		current := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		res = append(res, current.Val)
		if current.Right != nil {
			stack = append(stack, current.Right)
		}
		if current.Left != nil {
			stack = append(stack, current.Left)
		}
	}
	return res
}

// LC94 [left -> root -> right]
func InorderTraversal(root *TreeNode) []int {
	res := []int{}
	stack := []*TreeNode{}
	current := root
	for {
		for current != nil {
			stack = append(stack, current)
			current = current.Left
		}
		if len(stack) == 0 {
			break
		}
		current = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		res = append(res, current.Val)
		current = current.Right
	}
	return res
}

// LC145 [left -> right -> root]
func PostorderTraversal(root *TreeNode) []int {
	res := []int{}
	stack := []*TreeNode{root}
	for len(stack) > 0 {
		current := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		res = append([]int{current.Val}, res...)
		if current.Left != nil {
			stack = append(stack, current.Left)
		}
		if current.Right != nil {
			stack = append(stack, current.Right)
		}
	}
	return res
}

// LC 287 Floyd's cycle detection
func findDuplicate(nums []int) int {
	slow, fast := 0, 0
	slow = nums[slow]
	fast = nums[nums[fast]]
	for slow != fast {
		slow = nums[slow]
		fast = nums[nums[fast]]
	}
	// 2 pointers moving at the same speed
	slow = 0
	for slow != fast {
		slow = nums[slow]
		fast = nums[fast]
	}
	return slow
}

// LC173
type BSTIterator struct {
	stack []*TreeNode
}

func NewBSTIterator(root *TreeNode) BSTIterator {
	stack := []*TreeNode{}
	for root != nil {
		stack = append(stack, root)
		root = root.Left
	}
	return BSTIterator{stack: stack}
}

func (this *BSTIterator) Next() int {
	if len(this.stack) > 0 {
		current := this.stack[len(this.stack)-1]
		this.stack = this.stack[:len(this.stack)-1]
		val := current.Val
		current = current.Right
		for current != nil {
			this.stack = append(this.stack, current)
			current = current.Left
		}
		return val
	}
	return 0
}

func (this *BSTIterator) HasNext() bool {
	return len(this.stack) > 0
}
