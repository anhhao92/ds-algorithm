package problems

import (
	"container/list"
	"fmt"
	"slices"
	"strings"
)

func FindConnection(connections []string, name1 string, name2 string) int {
	// fred --- joe
	//		\ /
	//		mary
	//		 |
	//		bill
	type Node struct {
		name  string
		value int
	}
	connectionGraph := make(map[string][]string)
	for _, v := range connections {
		names := strings.Split(v, ":")
		connect1, connect2 := names[0], names[1]
		connectionGraph[connect1] = append(connectionGraph[connect1], connect2)
		connectionGraph[connect2] = append(connectionGraph[connect2], connect1)
	}
	// BFS
	visited := make(map[string]bool)
	queue := list.New()
	queue.PushBack(Node{name: name1, value: 0})
	for queue.Len() > 0 {
		node := queue.Front()
		name, value := node.Value.(Node).name, node.Value.(Node).value
		if name == name2 {
			return value
		}
		for _, v := range connectionGraph[name] {
			if !visited[v] {
				queue.PushBack(Node{name: v, value: value + 1})
			}
		}
		visited[name] = true
		queue.Remove(node)
	}
	return -1
}

// LC815
func NumBusesToDestination(routes [][]int, source int, target int) int {
	busRoutes := make(map[int][]int)
	for i := 0; i < len(routes); i++ {
		for _, v := range routes[i] {
			busRoutes[v] = append(busRoutes[v], i)
		}
	}
	visitedRoutes := make(map[int]bool)
	visitedBus := make(map[int]bool)
	queue := [][2]int{{source, 0}} // source, count

	for len(queue) > 0 {
		queue = queue[1:]
		src, count := queue[0][0], queue[0][1]
		if src == target {
			return count
		}
		visitedRoutes[src] = true
		for _, stop := range busRoutes[src] {
			if !visitedBus[stop] {
				visitedBus[stop] = true
				for _, v := range routes[stop] {
					if !visitedRoutes[v] {
						queue = append(queue, [2]int{v, count + 1})
					}
				}
			}
		}
	}
	return -1
}

type UnionFind struct {
	ranks           []int
	parents         []int
	numsOfComponent int
}

func NewUnionFind(n int) *UnionFind {
	parents := make([]int, n)
	ranks := make([]int, n)
	for i := 0; i < n; i++ {
		parents[i] = i
		ranks[i] = 1
	}
	return &UnionFind{parents: parents, ranks: ranks, numsOfComponent: n}
}

func (u *UnionFind) Find(n int) int {
	root := n
	for root != u.parents[root] {
		root = u.parents[root]
	}
	// Path compression
	for n != root {
		pre := u.parents[n]
		u.parents[n] = root
		n = pre
	}
	return root
}

func (u *UnionFind) Union(n1, n2 int) bool {
	p1, p2 := u.Find(n1), u.Find(n2)
	ranks, parents := u.ranks, u.parents
	if p1 == p2 {
		return false
	}
	if ranks[p1] > ranks[p2] {
		parents[p2] = p1
		ranks[p1] += ranks[p2]
	} else {
		parents[p1] = p2
		ranks[p2] += ranks[p1]
	}
	u.numsOfComponent--
	return true
}

// LC684
func findRedundantConnection(edges [][]int) []int {
	u := NewUnionFind(len(edges) + 1)
	for _, e := range edges {
		if !u.Union(e[0], e[1]) {
			return e
		}
	}
	return []int{}
}

func findCircleNum(isConnected [][]int) int {
	u := NewUnionFind(len(isConnected))
	for i := 0; i < len(isConnected); i++ {
		for j := 0; j < len(isConnected[i]); j++ {
			if i != j && isConnected[i][j] == 1 {
				u.Union(i, j)
			}
		}
	}
	return u.numsOfComponent
}

/*
*
LC 133
*/
type Node struct {
	Val       int
	Neighbors []*Node
}

func cloneGraph(node *Node) *Node {
	queue := []*Node{node}
	hs := map[*Node]*Node{node: {Val: node.Val}}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		curClone := hs[cur]
		for _, node := range cur.Neighbors {
			neightborClone := hs[node]
			if neightborClone == nil {
				queue = append(queue, node)
				neightborClone = &Node{Val: node.Val}
				hs[node] = neightborClone
			}
			curClone.Neighbors = append(curClone.Neighbors, neightborClone)
		}
	}
	return hs[node]
}

func LadderLength(beginWord string, endWord string, wordList []string) int {
	dict := make(map[string]bool)
	queue := []string{beginWord}
	count := 1
	for _, v := range wordList {
		dict[v] = true
	}
	for len(queue) > 0 {
		size := len(queue)
		for i := 0; i < size; i++ {
			word := queue[0]
			if word == endWord {
				return count
			}

			wordChars := []byte(word)
			for i := 0; i < len(wordChars); i++ {
				lastReplacementChar := wordChars[i]
				for j := byte('a'); j <= byte('z'); j++ {
					wordChars[i] = j
					replacedWord := string(wordChars)
					if dict[replacedWord] {
						queue = append(queue, replacedWord)
						delete(dict, replacedWord)
					}
				}
				wordChars[i] = lastReplacementChar
			}
			queue = queue[1:]
		}
		count++
	}
	return 0
}

// LC126
func FindLadders(beginWord string, endWord string, wordList []string) [][]string {
	dict := map[string]bool{}
	queue := []string{beginWord}
	visitedList := [][]string{}
	foundWord := false
	for _, v := range wordList {
		dict[v] = true
	}
	if !dict[endWord] {
		return [][]string{}
	}
	delete(dict, beginWord)

	isWordConnected := func(start, end string) bool {
		count := 0
		for i := 0; i < len(start) && count < 2; i++ {
			if start[i] != end[i] {
				count++
			}
		}
		return count == 1
	}

	for len(queue) > 0 && !foundWord {
		visitedList = append(visitedList, slices.Clone(queue))
		size := len(queue)
		for i := 0; i < size && !foundWord; i++ {
			word := queue[0]
			queue = queue[1:]
			for w := range dict {
				if isWordConnected(word, w) {
					if w == endWord {
						foundWord = true
						break
					}
					queue = append(queue, w)
					delete(dict, w)
				}
			}
		}
	}
	if !foundWord {
		return [][]string{}
	}
	result := [][]string{{endWord}}
	for i := len(visitedList) - 1; i >= 0; i-- {
		size := len(result)
		for j := 0; j < size; j++ {
			ans := result[0]
			result = result[1:]
			last := ans[0]
			for _, word := range visitedList[i] {
				if isWordConnected(last, word) {
					result = append(result, append([]string{word}, ans...))
				}
			}
		}
	}

	return result
}

func NumIslands(grid [][]byte) int {
	count := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == '1' {
				bfsIslands(i, j, grid)
				count++
			}
		}
	}
	return count
}
func bfsIslands(r, c int, grid [][]byte) {
	lenRow, lenCol := len(grid), len(grid[0])
	queue := [][]int{{r, c}}
	directions := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	grid[r][c] = '0'
	for len(queue) > 0 {
		row, col := queue[0][0], queue[0][1]
		for i := range directions {
			dr, dc := directions[i][0]+row, directions[i][1]+col
			if dr >= 0 && dr < lenRow && dc >= 0 && dc < lenCol && grid[dr][dc] == '1' {
				grid[dr][dc] = '0'
				queue = append(queue, []int{dr, dc})
			}
		}
		queue = queue[1:]
	}
}

func OrangesRotting(grid [][]int) int {
	countFreshOrange := 0
	queue := [][]int{}
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == 2 {
				queue = append(queue, []int{i, j})
			} else if grid[i][j] == 1 {
				countFreshOrange++
			}
		}
	}
	if countFreshOrange == 0 {
		return 0
	}
	timeElapse := bfsOrangesRotting(queue, grid, countFreshOrange)
	return timeElapse
}

func bfsOrangesRotting(queue [][]int, grid [][]int, countFreshOrange int) int {
	lenRow, lenCol := len(grid), len(grid[0])
	directions := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	level := 0
	for len(queue) > 0 {
		size := len(queue)
		for i := 0; i < size; i++ {
			row, col := queue[0][0], queue[0][1]
			queue = queue[1:]
			grid[row][col] = 0
			for i := range directions {
				dr, dc := directions[i][0]+row, directions[i][1]+col
				if dr >= 0 && dr < lenRow && dc >= 0 && dc < lenCol && grid[dr][dc] == 1 {
					grid[dr][dc] = 0
					countFreshOrange--
					queue = append(queue, []int{dr, dc})
				}
			}
		}
		level++
	}
	if countFreshOrange == 0 {
		return level - 1
	}
	return -1
}

func PacificAtlantic(heights [][]int) [][]int {
	rowLen, colLen := len(heights), len(heights[0])
	atlantic := make([][]bool, len(heights))
	pacific := make([][]bool, len(heights))
	result := [][]int{}
	for i := range heights {
		atlantic[i] = make([]bool, len(heights[0]))
		pacific[i] = make([]bool, len(heights[0]))
	}

	queue := addRow(0, [][]int{}, colLen)
	queue = addCol(0, queue, rowLen)
	bfsPacificAtlantic(queue, heights, atlantic)

	queue = addRow(rowLen-1, [][]int{}, colLen)
	queue = addCol(colLen-1, queue, rowLen)
	bfsPacificAtlantic(queue, heights, pacific)

	for i, row := range heights {
		for j := range row {
			if atlantic[i][j] && atlantic[i][j] == pacific[i][j] {
				result = append(result, []int{i, j})
			}
		}
	}
	return result
}

func addRow(row int, queue [][]int, totalCol int) [][]int {
	for col := 0; col < totalCol; col++ {
		queue = append(queue, []int{row, col})
	}
	return queue
}

func addCol(col int, queue [][]int, totalRow int) [][]int {
	for row := 0; row < totalRow; row++ {
		queue = append(queue, []int{row, col})
	}
	return queue
}

func bfsPacificAtlantic(queue [][]int, grid [][]int, table [][]bool) {
	visited := map[string]bool{}
	directions := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	lenRow, lenCol := len(grid), len(grid[0])

	for len(queue) > 0 {
		row, col := queue[0][0], queue[0][1]
		queue = queue[1:]
		if visited[fmt.Sprint(row, col)] {
			continue
		}
		table[row][col] = true
		visited[fmt.Sprint(row, col)] = true
		for i := range directions {
			dr, dc := directions[i][0]+row, directions[i][1]+col
			if dr >= 0 && dr < lenRow && dc >= 0 && dc < lenCol &&
				!visited[fmt.Sprint(dr, dc)] && grid[row][col] <= grid[dr][dc] {
				queue = append(queue, []int{dr, dc})
			}
		}
	}
}

func solve(board [][]byte) {
	row, col := len(board), len(board[0])
	for i := 0; i < row; i++ {
		for j := 0; j < col; {
			if board[i][j] == 'O' {
				board[i][j] = 'T'
				bfsORegion([][]int{{i, j}}, board)
			}
			if i > 0 && i < row-1 {
				j += col - 1
			} else {
				j++
			}
		}
	}
	for i, row := range board {
		for j, col := range row {
			if col == 'O' {
				board[i][j] = 'X'
			} else if col == 'T' {
				board[i][j] = 'O'
			}
		}
	}
}

func bfsORegion(queue [][]int, board [][]byte) {
	directions := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	lenRow, lenCol := len(board), len(board[0])

	for len(queue) > 0 {
		row, col := queue[0][0], queue[0][1]
		queue = queue[1:]
		for i := range directions {
			dr, dc := directions[i][0]+row, directions[i][1]+col
			if dr >= 0 && dr < lenRow && dc >= 0 && dc < lenCol && board[dr][dc] == 'O' {
				board[dr][dc] = 'T'
				queue = append(queue, []int{dr, dc})
			}
		}
	}
}

// Kahn's algorithm: Leetcode 207
// Indegree: incoming edge
// Add node's indegree=0 to queue -> visit node -> remove all edge -> continue adding indegree node = 0 to queue
func CanFinish(numCourses int, prerequisites [][]int) bool {
	indegree := make([]int, numCourses)
	nodes := make([][]int, numCourses)
	queue := []int{}
	visited := 0
	// 2D slices neightbor list
	// [0, 1] <=> 0 <- 1
	for _, row := range prerequisites {
		out, incoming := row[1], row[0]
		nodes[out] = append(nodes[out], incoming)
		indegree[incoming]++
	}
	for i := range numCourses {
		if indegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		visited++
		// remove all indegree to current node
		for _, next := range nodes[current] {
			indegree[next]--
			if indegree[next] == 0 {
				queue = append(queue, next)
			}
		}
	}
	return visited == numCourses
}

// Topological Sort: Leetcode 210
func FindOrder(numCourses int, prerequisites [][]int) []int {
	indegree := make([]int, numCourses)
	nodes := make([][]int, numCourses)
	queue, result := []int{}, []int{}
	for _, row := range prerequisites {
		out, incoming := row[1], row[0]
		nodes[out] = append(nodes[out], incoming)
		indegree[incoming]++
	}
	for i := range numCourses {
		if indegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		result = append(result, current)
		for _, next := range nodes[current] {
			indegree[next]--
			if indegree[next] == 0 {
				queue = append(queue, next)
			}
		}
	}
	if len(result) == numCourses {
		return result
	}
	return []int{}
}

// LC 2569
func checkValidGridKnightConfiguration(grid [][]int) bool {
	n := len(grid)
	moves := [][2]int{{-2, 1}, {-2, -1}, {-1, 2}, {-1, -2}, {2, 1}, {2, -1}, {1, 2}, {1, -2}}
	nextRow, nextCol := 0, 0
	for i := 0; i < n*n; i++ {
		if grid[nextRow][nextCol] != i {
			return false
		}
		for _, move := range moves {
			r, c := nextRow+move[0], nextCol+move[1]
			if r >= 0 && r < n && c >= 0 && c < n && grid[r][c] == i+1 {
				nextRow, nextCol = r, c
			}
		}
	}
	return true
}
