package problems

import (
	"fmt"
	"math"
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
	var prev *TreeNode
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
		if prev != nil && prev.Val >= cur.Val {
			return false
		}
		prev = cur
		right := dfs(cur.Right)
		return right
	}
	return dfs(root)
}

// LC 958
func isCompleteTree(root *TreeNode) bool {
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		q := queue[0]
		queue = queue[1:]
		if q != nil {
			queue = append(queue, q.Left)
			queue = append(queue, q.Right)
		} else {
			for len(queue) > 0 {
				node := queue[0]
				queue = queue[1:]
				if node != nil {
					return false
				}
			}
		}
	}
	return true
}

// LC 662
func widthOfBinaryTree(root *TreeNode) int {
	queue := [][3]any{{root, 1, 0}}
	res := 0
	prevNum, prevLevel := 1, 0
	for len(queue) > 0 {
		q := queue[0]
		node, num, level := q[0].(*TreeNode), q[1].(int), q[2].(int)
		queue = queue[1:]
		if level > prevLevel {
			prevLevel = level
			prevNum = num
		}
		res = max(res, num-prevNum+1)
		if node.Left != nil {
			queue = append(queue, [3]any{node.Left, num * 2, level + 1})
		}
		if node.Right != nil {
			queue = append(queue, [3]any{node.Right, num*2 + 1, level + 1})
		}
	}
	return res
}

// LC 513
func findBottomLeftValue(root *TreeNode) int {
	queue := []*TreeNode{root}
	current := root
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		// visit right first
		if current.Right != nil {
			queue = append(queue, current.Right)
		}
		if current.Left != nil {
			queue = append(queue, current.Left)
		}
	}
	return current.Val
}

// LC 669
func trimBST(root *TreeNode, low int, high int) *TreeNode {
	if root == nil {
		return root
	}
	if root.Val > high {
		return trimBST(root.Left, low, high)
	}
	if root.Val < low {
		return trimBST(root.Right, low, high)
	}
	root.Left = trimBST(root.Left, low, high)
	root.Right = trimBST(root.Right, low, high)
	return root
}

// LC 872
func leafSimilar(root1 *TreeNode, root2 *TreeNode) bool {
	var dfs func(r *TreeNode, leaf *[]int)
	dfs = func(r *TreeNode, leaf *[]int) {
		if r == nil {
			return
		}
		if r.Left == nil && r.Right == nil {
			*leaf = append(*leaf, r.Val)
			return
		}
		dfs(r.Left, leaf)
		dfs(r.Right, leaf)
	}
	l1, l2 := []int{}, []int{}
	dfs(root1, &l1)
	dfs(root2, &l2)
	return slices.Equal(l1, l2)
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

// LC 101
func isSymmetric(root *TreeNode) bool {
	var dfs func(l *TreeNode, r *TreeNode) bool
	dfs = func(l *TreeNode, r *TreeNode) bool {
		if l == nil && r == nil {
			return true
		}
		if l == nil || r == nil {
			return false
		}
		return l.Val == r.Val && dfs(l.Left, r.Right) && dfs(l.Right, r.Left)
	}
	return dfs(root.Left, root.Right)
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

// LC 230
func kthSmallest(root *TreeNode, k int) int {
	stack := []*TreeNode{}
	cur := root
	// inorder traversal
	for root != nil {
		// go to the most left
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		if len(stack) == 0 {
			break
		}
		// visit root node
		cur = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if k == 1 {
			return cur.Val
		}
		k--
		// visit right node
		cur = cur.Right
	}
	return -1
}

// LC 105
func buildTreeFromPreorderAndInorder(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 || len(inorder) == 0 {
		return nil
	}
	root := &TreeNode{Val: preorder[0]}
	mid := slices.Index(inorder, preorder[0])
	root.Left = buildTreeFromPreorderAndInorder(preorder[1:mid+1], inorder[:mid])
	root.Right = buildTreeFromPreorderAndInorder(preorder[mid+1:], inorder[mid+1:])
	return root
}

// LC 106
func buildTree(inorder []int, postorder []int) *TreeNode {
	inorderValToIndex := map[int]int{}
	for i, v := range inorder {
		inorderValToIndex[v] = i
	}
	var build func(l, r int) *TreeNode
	build = func(l, r int) *TreeNode {
		if l > r {
			return nil
		}
		root := &TreeNode{Val: postorder[len(postorder)-1]}
		index := inorderValToIndex[root.Val]
		postorder = postorder[:len(postorder)-1]
		root.Right = build(index+1, r)
		root.Left = build(l, index-1)
		return root
	}
	return build(0, len(inorder)-1)
}

// LC 652
func findDuplicateSubtrees(root *TreeNode) []*TreeNode {
	hashMap := map[string]int{}
	res := []*TreeNode{}
	var dfs func(r *TreeNode) string
	dfs = func(r *TreeNode) string {
		if r == nil {
			return "x"
		}
		s := fmt.Sprintf("%v,%s,%s", r.Val, dfs(r.Left), dfs(r.Right))
		if hashMap[s] == 1 {
			res = append(res, r)
		}
		hashMap[s]++
		return s
	}
	dfs(root)
	return res
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

// LC 199
func rightSideView(root *TreeNode) []int {
	queue := []*TreeNode{root}
	res := []int{}
	for len(queue) > 0 {
		var rightNode *TreeNode
		for _, node := range queue {
			queue = queue[1:]
			if node != nil {
				rightNode = node
				queue = append(queue, node.Left)
				queue = append(queue, node.Right)
			}
		}
		if rightNode != nil {
			res = append(res, rightNode.Val)
		}
	}
	return res
}

// LC 1448
func goodNodes(root *TreeNode) int {
	var dfs func(node *TreeNode, prev int) int
	dfs = func(node *TreeNode, prev int) int {
		if node == nil {
			return 0
		}
		count := 0
		if node.Val >= prev {
			count = 1
			prev = node.Val
		}
		return dfs(node.Left, prev) + dfs(node.Right, prev) + count
	}
	return dfs(root, root.Val)
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

// LC 337
func RobHouseIII(root *TreeNode) int {
	var dfs func(r *TreeNode) (int, int) // withRoot, withoutRoot
	dfs = func(r *TreeNode) (int, int) {
		if r == nil {
			return 0, 0
		}
		left, withoutLeft := dfs(r.Left)
		right, withoutRight := dfs(r.Right)
		withRoot := r.Val + withoutLeft + withoutRight
		withoutRoot := max(left, withoutLeft) + max(right, withoutRight)
		return withRoot, withoutRoot
	}
	return max(dfs(root))
}

// LC 129
func sumNumbers(root *TreeNode) int {
	var dfs func(r *TreeNode, val int) int
	dfs = func(r *TreeNode, val int) int {
		if r == nil {
			return 0
		}
		sum := val*10 + r.Val
		if r.Left == nil && r.Right == nil {
			return sum
		}
		return dfs(r.Left, sum) + dfs(r.Right, sum)
	}
	return dfs(root, 0)
}

// LC 95
func GenerateUniqueTrees(n int) []*TreeNode {
	var generate func(l, r int) []*TreeNode
	generate = func(l, r int) []*TreeNode {
		if l > r {
			return []*TreeNode{nil}
		}
		res := []*TreeNode{}
		for val := l; val <= r; val++ {
			for _, leftTree := range generate(l, val-1) {
				for _, rightTree := range generate(val+1, r) {
					root := &TreeNode{Val: val, Left: leftTree, Right: rightTree}
					res = append(res, root)
				}
			}
		}
		return res
	}
	return generate(1, n)
}

// LC 894
func allPossibleFBT(n int) []*TreeNode {
	dp := map[int][]*TreeNode{}
	var dfs func(num int) []*TreeNode
	dfs = func(num int) []*TreeNode {
		if num%2 == 0 {
			return []*TreeNode{}
		}
		if num == 1 {
			return []*TreeNode{{Val: 0}}
		}
		if v, ok := dp[num]; ok {
			return v
		}
		res := []*TreeNode{}
		for i := 1; i < num; i++ {
			leftTrees := dfs(i)
			rightTrees := dfs(num - i - 1)
			for _, l := range leftTrees {
				for _, r := range rightTrees {
					root := &TreeNode{Val: 0, Left: l, Right: r}
					res = append(res, root)
				}
			}
		}
		dp[n] = res
		return res
	}
	return dfs(n)
}

// LC 951
func flipEquiv(root1 *TreeNode, root2 *TreeNode) bool {
	if root1 == nil && root2 == nil {
		return true
	}
	// Just one of the trees is empty
	if root1 == nil || root2 == nil {
		return false
	}
	if root1.Val != root2.Val {
		return false
	}
	if flipEquiv(root1.Left, root2.Left) && flipEquiv(root1.Right, root2.Right) {
		return true
	}
	return flipEquiv(root1.Left, root2.Right) && flipEquiv(root1.Right, root2.Left)
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

// LC 606
func tree2str(root *TreeNode) string {
	if root == nil {
		return ""
	}
	left := tree2str(root.Left)
	right := tree2str(root.Right)

	if left == "" && right == "" {
		return fmt.Sprintf("%d", root.Val)
	}
	if right == "" {
		return fmt.Sprintf("%d(%s)", root.Val, left)
	}
	return fmt.Sprintf("%d(%s)(%s)", root.Val, left, right)
}

// LC 1457
func pseudoPalindromicPaths(root *TreeNode) int {
	odd := 0
	count := [10]int{}
	var dfs func(r *TreeNode) int
	dfs = func(r *TreeNode) int {
		if r == nil {
			return 0
		}
		count[r.Val]++
		oddChange := -1
		if count[r.Val]%2 == 1 {
			oddChange = 1
		}
		odd += oddChange
		res := 0
		if r.Left == nil && r.Right == nil {
			if odd <= 1 {
				res = 1
			}
		} else {
			res = dfs(r.Left) + dfs(r.Right)
		}
		odd -= oddChange
		count[r.Val]--
		return res
	}
	return dfs(root)
}

// LC 1361
func validateBinaryTreeNodes(n int, leftChild []int, rightChild []int) bool {
	hasParent := map[int]bool{}
	for _, v := range leftChild {
		if v != -1 {
			hasParent[v] = true
		}
	}
	for _, v := range rightChild {
		if v != -1 {
			hasParent[v] = true
		}
	}
	if len(hasParent) == n {
		return false
	}
	root := -1
	for i := range n {
		if _, ok := hasParent[i]; !ok {
			root = i
			break
		}
	}
	visited := map[int]bool{}
	var dfs func(i int) bool
	dfs = func(i int) bool {
		if i == -1 {
			return true
		}
		if visited[i] {
			return false
		}
		visited[i] = true
		return dfs(leftChild[i]) && dfs(rightChild[i])
	}
	return dfs(root) && len(visited) == n
}

// LC 1325
func removeLeafNodes(root *TreeNode, target int) *TreeNode {
	if root == nil {
		return nil
	}
	// Postorder traversal
	root.Left = removeLeafNodes(root.Left, target)
	root.Right = removeLeafNodes(root.Right, target)
	if root.Left == nil && root.Right == nil && root.Val == target {
		return nil
	}
	return root
}

// LC 538
func convertBST(root *TreeNode) *TreeNode {
	sum := 0
	var dfs func(r *TreeNode)
	dfs = func(r *TreeNode) {
		if r == nil {
			return
		}
		// right - root - left
		dfs(r.Right)
		tmp := r.Val
		r.Val += sum
		sum += tmp
		dfs(r.Left)
	}
	dfs(root)
	return root
}

// LC 979
func distributeCoins(root *TreeNode) int {
	res := 0
	var dfs func(r *TreeNode) (int, int)
	dfs = func(r *TreeNode) (int, int) {
		if r == nil {
			return 0, 0
		}
		leftSize, leftCoin := dfs(r.Left)
		rightSize, rightCoin := dfs(r.Right)
		size := 1 + leftSize + rightSize
		coin := r.Val + leftCoin + rightCoin
		res += int(math.Abs(float64(size - coin)))
		return size, coin
	}
	dfs(root)
	return res
}

// LC 988
func smallestFromLeaf(root *TreeNode) string {
	var dfs func(r *TreeNode, prev string) string
	dfs = func(r *TreeNode, prev string) string {
		if r == nil {
			return ""
		}
		prev = string(r.Val+'a') + prev
		if r.Left != nil && r.Right != nil {
			left := dfs(r.Left, prev)
			right := dfs(r.Right, prev)
			return min(left, right)
		}
		if r.Right != nil {
			return dfs(r.Right, prev)
		}
		if r.Left != nil {
			return dfs(r.Left, prev)
		}
		return prev
	}
	return dfs(root, "")
}

// LC 1609
func isEvenOddTree(root *TreeNode) bool {
	queue := []*TreeNode{root}
	isOdd := true
	for len(queue) > 0 {
		prev := math.MaxInt32
		if isOdd {
			prev = math.MinInt32
		}
		for _, q := range queue {
			if isOdd && (q.Val%2 == 0 || q.Val <= prev) {
				return false
			} else if !isOdd && (q.Val%2 == 1 || q.Val >= prev) {
				return false
			}
			if q.Left != nil {
				queue = append(queue, q.Left)
			}
			if q.Right != nil {
				queue = append(queue, q.Right)
			}
			queue = queue[1:]
			prev = q.Val
		}
		isOdd = !isOdd
	}
	return true
}
