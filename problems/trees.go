package problems

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
	Next  *TreeNode
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

// LC 450
func deleteNode(root *TreeNode, key int) *TreeNode {
	if root == nil {
		return root
	}
	if key > root.Val {
		root.Right = deleteNode(root.Right, key)
	} else if key < root.Val {
		root.Left = deleteNode(root.Left, key)
	} else {
		if root.Left == nil {
			return root.Right
		}
		if root.Right == nil {
			return root.Left
		}
		// find the smallest on the right
		cur := root.Right
		for cur.Left != nil {
			cur = cur.Left
		}
		root.Val = cur.Val // root, cur at same value
		root.Right = deleteNode(root.Right, root.Val)
	}
	return root
}

// LC 701
func insertIntoBST(root *TreeNode, val int) *TreeNode {
	if root == nil {
		return &TreeNode{Val: val}
	}
	cur := root
	for cur != nil {
		if val > cur.Val {
			if cur.Right == nil {
				cur.Right = &TreeNode{Val: val}
				return root
			}
			cur = cur.Right
		} else {
			if cur.Left == nil {
				cur.Left = &TreeNode{Val: val}
				return root
			}
			cur = cur.Left
		}
	}
	return root
	// if val > root.Val {
	// 	root.Right = insertIntoBST(root.Right, val)
	// } else {
	// 	root.Left = insertIntoBST(root.Left, val)
	// }
}

// LC 226
func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return root
	}
	root.Left, root.Right = root.Right, root.Left
	invertTree(root.Left)
	invertTree(root.Right)
	return root
}

// LC 543
func diameterOfBinaryTree(root *TreeNode) int {
	maxVal := 0
	var dfs func(current *TreeNode) int
	dfs = func(current *TreeNode) int {
		if current == nil {
			return 0
		}
		left := dfs(current.Left)
		right := dfs(current.Right)
		maxVal = max(maxVal, left+right)
		return max(left, right) + 1
	}
	dfs(root)
	return maxVal
}

// LC 110
func isBalancedTree(root *TreeNode) bool {
	var dfs func(r *TreeNode) (bool, int)
	dfs = func(r *TreeNode) (bool, int) {
		if r == nil {
			return true, 0
		}
		left, leftHeight := dfs(r.Left)
		right, rightHeight := dfs(r.Right)
		isBalanced := left && right && abs(leftHeight-rightHeight) <= 1
		return isBalanced, max(leftHeight, rightHeight) + 1
	}
	res, _ := dfs(root)
	return res
}

// LC 100
func isSameTree(p *TreeNode, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}
	if !(p != nil && q != nil) {
		return false
	}
	if p.Val != q.Val {
		return false
	}
	return isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
}

// LC 572
func isSubtree(root *TreeNode, subRoot *TreeNode) bool {
	if root == nil {
		return false
	}
	if subRoot == nil {
		return true
	}
	if isSameTree(root, subRoot) {
		return true
	}
	return isSubtree(root.Left, subRoot) || isSubtree(root.Right, subRoot)
}

// LC 108
func sortedArrayToBST(nums []int) *TreeNode {
	var dfs func(l, r int) *TreeNode
	dfs = func(l, r int) *TreeNode {
		if l > r {
			return nil
		}
		mid := (l + r) / 2
		root := &TreeNode{Val: nums[mid]}
		root.Left = dfs(l, mid-1)
		root.Right = dfs(mid+1, r)
		return root
	}
	return dfs(0, len(nums)-1)
}

// LC 617
func mergeTrees(root1 *TreeNode, root2 *TreeNode) *TreeNode {
	if root1 == nil && root2 == nil {
		return nil
	}
	v1, v2 := 0, 0
	if root1 != nil {
		v1 = root1.Val
	}
	if root2 != nil {
		v2 = root2.Val
	}
	root := &TreeNode{Val: v1 + v2}
	var left1, left2, right1, right2 *TreeNode
	if root1 != nil {
		left1 = root1.Left
		right1 = root1.Right
	}
	if root2 != nil {
		left2 = root2.Left
		right2 = root2.Right
	}
	root.Left = mergeTrees(left1, left2)
	root.Right = mergeTrees(right1, right2)
	return root
}

// LC 116/117
func connectNextRightToEachNode(root *TreeNode) *TreeNode {
	if root == nil {
		return root
	}
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		// pre will be reset to nil after completed each level
		var pre *TreeNode
		var size = len(queue)
		for i := 0; i < size; i++ {
			current := queue[i]
			if pre != nil {
				pre.Next = current
			}
			pre = current
			if current.Left != nil {
				queue = append(queue, current.Left)
			}
			if current.Right != nil {
				queue = append(queue, current.Right)
			}
		}
		queue = queue[size:]
	}
	return root
}

// LC 235
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	cur := root
	for cur != nil {
		if p.Val > cur.Val && q.Val > cur.Val {
			cur = cur.Right
		} else if p.Val < cur.Val && q.Val < cur.Val {
			cur = cur.Left
		} else {
			return cur
		}
	}
	return cur
}

// LC 103
func ZigzagLevelOrder(root *TreeNode) [][]int {
	res := [][]int{}
	if root == nil {
		return res
	}
	queue := []*TreeNode{root}
	isLeftToRight := true
	for len(queue) > 0 {
		var size = len(queue)
		values := []int{}
		for _, current := range queue {
			values = append(values, current.Val)
			if current.Left != nil {
				queue = append(queue, current.Left)
			}
			if current.Right != nil {
				queue = append(queue, current.Right)
			}
		}
		if !isLeftToRight {
			slices.Reverse(values)
		}
		queue = queue[size:]
		res = append(res, values)
		isLeftToRight = !isLeftToRight
	}
	return res
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
