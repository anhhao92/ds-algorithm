package problems

import (
	"fmt"
	"math"
	"slices"
	"sort"
)

// LC 332
func FindItinerary(tickets [][]string) []string {
	adj := make(map[string][]string)
	ans := []string{}

	// Fill the adjacency list
	for i := 0; i < len(tickets); i++ {
		adj[tickets[i][0]] = append(adj[tickets[i][0]], tickets[i][1])
	}

	// Sort the destinations in lexical order
	for key := range adj {
		sort.Strings(adj[key])
	}

	// Use a stack to store the itinerary
	stack := []string{"JFK"}

	for len(stack) > 0 {
		src := stack[len(stack)-1]
		if len(adj[src]) == 0 {
			ans = append(ans, src)
			stack = stack[:len(stack)-1]
		} else {
			dst := adj[src][0]
			adj[src] = adj[src][1:]
			stack = append(stack, dst)
		}
	}
	// Reverse the answer to get the correct order
	slices.Reverse(ans)
	return ans
}

// LC 463
func islandPerimeter(grid [][]int) int {
	directions := [4][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	m, n := len(grid), len(grid[0])
	var dfs func(r, c int) int
	dfs = func(r, c int) int {
		count := 0
		grid[r][c] = 2
		for _, v := range directions {
			nr, nc := r+v[0], c+v[1]
			if min(nr, nc) < 0 || nr == m || nc == n || grid[nr][nc] == 0 {
				count++
			} else if grid[nr][nc] == 1 {
				count += dfs(nr, nc)
			}
		}
		return count
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] == 1 {
				return dfs(i, j)
			}
		}
	}
	return 0
}

func isAlienSorted(words []string, order string) bool {
	var lexicalOrder [26]int
	for i := 0; i < len(order); i++ {
		lexicalOrder[order[i]-'a'] = i
	}
	isLessThanEqual := func(w1, w2 string) bool {
		n := min(len(w1), len(w2))
		for i := 0; i < n; i++ {
			c1, c2 := w1[i]-'a', w2[i]-'a'
			if c1 != c2 {
				return lexicalOrder[c1] < lexicalOrder[c2]
			}
		}
		return true
	}
	for i := 0; i < len(words)-1; i++ {
		if !isLessThanEqual(words[i], words[i+1]) {
			return false
		}
	}
	return true
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

// LC 721
func accountsMerge(accounts [][]string) [][]string {
	uf := NewUnionFind(len(accounts))
	emailToAccount := map[string]int{}
	for i, acc := range accounts {
		for _, email := range acc[1:] {
			if idx, ok := emailToAccount[email]; ok {
				uf.Union(i, idx)
			} else {
				emailToAccount[email] = i
			}
		}
	}
	groups := map[int][]string{}
	for k, v := range emailToAccount {
		leader := uf.Find(v)
		groups[leader] = append(groups[leader], k)
	}
	res := [][]string{}
	for k := range groups {
		name := accounts[k][0]
		g := groups[k]
		slices.Sort(g)
		res = append(res, append([]string{name}, g...))
	}
	return res
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

// LC 1905
func countSubIslands(grid1 [][]int, grid2 [][]int) int {
	row, col := len(grid1), len(grid1[0])
	visited := make([][]bool, row)
	for i := range visited {
		visited[i] = make([]bool, col)
	}
	var dfs func(i, j int) bool
	dfs = func(i, j int) bool {
		if i < 0 || j < 0 || i == row || j == col || grid2[i][j] == 0 || visited[i][j] {
			return true
		}
		visited[i][j] = true
		res := true
		if grid1[i][j] == 0 {
			res = false
		}
		res = dfs(i-1, j) && res
		res = dfs(i+1, j) && res
		res = dfs(i, j-1) && res
		res = dfs(i, j+1) && res
		return res
	}
	res := 0
	for i := 0; i < row; i++ {
		for j := 0; j < col; j++ {
			if grid2[i][j] == 1 && !visited[i][j] && dfs(i, j) {
				res++
			}
		}
	}
	return res
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

// LC 1462
func checkIfPrerequisite(numCourses int, prerequisites [][]int, queries [][]int) []bool {
	indegree := make([]int, numCourses)
	adj := make([][]int, numCourses)
	for _, row := range prerequisites {
		src, dest := row[0], row[1]
		adj[src] = append(adj[src], dest)
		indegree[src]++
	}
	queue := []int{}
	for i := range indegree {
		if indegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	isReachable := make([][]bool, numCourses)
	for i := 0; i < numCourses; i++ {
		isReachable[i] = make([]bool, numCourses)
	}
	for len(queue) > 0 {
		src := queue[0]
		queue = queue[1:]
		for _, next := range adj[src] {
			isReachable[next][src] = true
			// find all prerequisites src
			for i := 0; i < numCourses; i++ {
				if isReachable[src][i] {
					isReachable[next][i] = true
				}
			}
			indegree[next]--
			if indegree[next] == 0 {
				queue = append(queue, next)
			}
		}
	}
	res := make([]bool, len(queries))
	for i, v := range queries {
		src, dst := v[0], v[1]
		res[i] = isReachable[dst][src]
	}
	return res
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

// LC 1857 Topological sort
func largestPathValue(colors string, edges [][]int) int {
	n := len(colors)
	adj := make([][]int, n)
	indegree := make([]int, n)
	for _, v := range edges {
		src, dest := v[0], v[1]
		adj[src] = append(adj[src], dest)
		indegree[dest]++
	}
	queue := []int{}
	for i := range indegree {
		if indegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	count := make([][26]int, n)
	res, visited := 0, 0
	for len(queue) > 0 {
		src := queue[0]
		queue = queue[1:]
		visited++
		count[src][colors[src]-'a']++
		res = max(res, count[src][colors[src]-'a'])
		for _, neightbor := range adj[src] {
			for c := range count[src] {
				count[neightbor][c] = max(count[neightbor][c], count[src][c])
			}
			indegree[neightbor]--
			if indegree[neightbor] == 0 {
				queue = append(queue, neightbor)
			}
		}
	}
	if visited == n {
		return res
	}
	return -1
}

// LC 2050
func minimumTime(n int, relations [][]int, time []int) int {
	adj := make([][]int, n)
	indegree := make([]int, n)
	for _, v := range relations {
		src, dest := v[0], v[1]
		adj[src] = append(adj[src], dest)
		indegree[dest]++
	}
	queue := []int{}
	maxTime := make([]int, n)
	for i := range indegree {
		if indegree[i] == 0 {
			queue = append(queue, i)
			maxTime[i] = time[i]
		}
	}
	for len(queue) > 0 {
		src := queue[0]
		queue = queue[1:]
		for _, neightbor := range adj[src] {
			maxTime[neightbor] = max(maxTime[neightbor], maxTime[src]+time[neightbor])
			indegree[neightbor]--
			if indegree[neightbor] == 0 {
				queue = append(queue, neightbor)
			}
		}
	}
	return slices.Max(maxTime)
}

// LC 802
func eventualSafeNodes(graph [][]int) []int {
	n := len(graph)
	indegree := make([]int, n)
	adj := make([][]int, n)
	// reverse edges
	for i := 0; i < n; i++ {
		for _, node := range graph[i] {
			adj[node] = append(adj[node], i)
			indegree[i]++
		}
	}
	queue := []int{}
	for i := 0; i < n; i++ {
		if indegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	res := []int{}
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		res = append(res, node)
		for _, v := range adj[node] {
			indegree[v]--
			if indegree[v] == 0 {
				queue = append(queue, v)
			}
		}
	}
	return res
}

// LC 1219
func getMaximumGold(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	directions := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}

	var dfs func(r, c int) int
	dfs = func(r, c int) int {
		if min(r, c) < 0 || r == m || c == n || grid[r][c] == 0 {
			return 0
		}
		gold := grid[r][c]
		grid[r][c] = 0
		res := grid[r][c]
		for _, v := range directions {
			nr, nc := r+v[0], c+v[1]
			res = max(res, grid[r][c]+dfs(nr, nc))
		}
		grid[r][c] = gold
		return res
	}
	res := 0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] != 0 {
				res = max(res, dfs(i, j))
			}
		}
	}
	return res
}

func findAllPeople(n int, meetings [][]int, firstPerson int) []int {
	adj := map[int][][2]int{}
	for _, v := range meetings {
		x, y, t := v[0], v[1], v[2]
		adj[x] = append(adj[x], [2]int{y, t})
		adj[y] = append(adj[y], [2]int{x, t})
	}
	earliest := make([]int, n)
	for i := range earliest {
		earliest[i] = math.MaxInt32
	}
	earliest[0] = 0
	earliest[firstPerson] = 0
	queue := [][2]int{{0, 0}, {firstPerson, 0}}
	for len(queue) > 0 {
		person, time := queue[0][0], queue[0][1]
		queue = queue[1:]
		for _, neightbor := range adj[person] {
			nextPerson, t := neightbor[0], neightbor[1]
			if t >= time && earliest[nextPerson] > t {
				earliest[nextPerson] = t
				queue = append(queue, [2]int{nextPerson, t})
			}
		}
	}
	res := []int{}
	for i, v := range earliest {
		if v != math.MaxInt32 {
			res = append(res, i)
		}
	}
	return res
}

// LC 1466
func minReorder(n int, connections [][]int) int {
	adj := map[int][]int{}
	edges := map[[2]int]bool{}
	for _, v := range connections {
		a, b := v[0], v[1]
		adj[a] = append(adj[a], b)
		adj[b] = append(adj[b], a)
		edges[[2]int{a, b}] = true
	}
	changes := 0
	visited := map[int]bool{}
	var dfs func(city int)
	dfs = func(city int) {
		visited[city] = true
		for _, neighbor := range adj[city] {
			if visited[neighbor] {
				continue
			}
			e := [2]int{neighbor, city}
			if !edges[e] {
				changes++
			}
			dfs(neighbor)
		}
	}
	dfs(0)
	return changes
}

// LC 909
func snakesAndLadders(board [][]int) int {
	n := len(board)
	queue := [][2]int{{1, 0}} // [square, move]
	visited := map[int]bool{}
	slices.Reverse(board)
	getPosition := func(num int) (r int, c int) {
		r = (num - 1) / n
		c = (num - 1) % n
		if r%2 != 0 {
			c = n - 1 - c
		}
		return
	}

	for len(queue) > 0 {
		square, move := queue[0][0], queue[0][1]
		queue = queue[1:]
		for i := 1; i <= 6; i++ {
			next := square + i
			r, c := getPosition(next)
			if board[r][c] != -1 {
				next = board[r][c]
			}
			if next == n*n {
				return move + 1
			}
			if !visited[next] {
				visited[next] = true
				queue = append(queue, [2]int{next, move + 1})
			}
		}
	}
	return -1
}

// LC 752
func OpenLock(deadends []string, target string) int {
	deadendsSet := map[string]bool{}
	for _, v := range deadends {
		deadendsSet[v] = true
	}
	if deadendsSet["0000"] {
		return -1
	}
	generateLock := func(lock []byte) (res []string) {
		for i := 0; i < 4; i++ {
			character := lock[i] - '0'
			lock[i] = ((character + 1) % 10) + '0'
			res = append(res, string(lock))
			lock[i] = ((character - 1 + 10) % 10) + '0'
			res = append(res, string(lock))
			lock[i] = character + '0'
		}
		return res
	}

	queue := [][]byte{{'0', '0', '0', '0', 0}} // [0,0,0,0, move]
	for len(queue) > 0 {
		lock, move := queue[0][:4], queue[0][4]
		queue = queue[1:]
		if string(lock) == target {
			return int(move)
		}
		for _, v := range generateLock(lock) {
			if v == target {
				return int(move) + 1
			}
			if !deadendsSet[v] {
				deadendsSet[v] = true
				queue = append(queue, append([]byte(v), move+1))
			}
		}
	}
	return -1
}

// LC 934
func ShortestBridge(grid [][]int) int {
	n := len(grid)
	directions := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	queue := [][2]int{} // [row, col]
	visited := make([][]bool, n)
	for i := range visited {
		visited[i] = make([]bool, n)
	}
	var dfs func(r, c int)
	dfs = func(r, c int) {
		queue = append(queue, [2]int{r, c})
		visited[r][c] = true
		for _, v := range directions {
			nr, nc := r+v[0], c+v[1]
			if nr >= 0 && nr < n && nc >= 0 && nc < n && !visited[nr][nc] && grid[nr][nc] == 1 {
				dfs(nr, nc)
			}
		}
	}
	bfs := func() int {
		level := 0
		for len(queue) > 0 {
			size := len(queue)
			for _, q := range queue {
				r, c := q[0], q[1]
				for _, v := range directions {
					nr, nc := r+v[0], c+v[1]
					if nr >= 0 && nr < n && nc >= 0 && nc < n && !visited[nr][nc] {
						if grid[nr][nc] == 1 {
							return level
						}
						queue = append(queue, [2]int{nr, nc})
						visited[nr][nc] = true
					}
				}
			}
			level++
			queue = queue[size:]
		}
		return level
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] == 1 {
				dfs(i, j)
				return bfs()
			}
		}
	}
	return 0
}

// LC 1091
func shortestPathBinaryMatrix(grid [][]int) int {
	n := len(grid)
	if grid[0][0] == 1 || grid[n-1][n-1] == 1 {
		return -1
	}
	directions := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}}
	queue := [][3]int{{0, 0, 1}} // [row, col, count]
	grid[0][0] = 1               // mark 1st move as visited
	for len(queue) > 0 {
		q := queue[0]
		queue = queue[1:]
		r, c, count := q[0], q[1], q[2]
		if r == n-1 && c == n-1 {
			return count
		}
		for _, v := range directions {
			nr, nc := r+v[0], c+v[1]
			if nr >= 0 && nr < n && nc >= 0 && nc < n && grid[nr][nc] == 0 {
				queue = append(queue, [3]int{nr, nc, count + 1})
				grid[nr][nc] = 1
			}
		}
	}
	return -1
}

// LC 1162
func maxDistance(grid [][]int) int {
	n := len(grid)
	queue := [][2]int{} // [row, col]
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] == 1 {
				queue = append(queue, [2]int{i, j})
			}
		}
	}
	res := -1
	directions := [4][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	for len(queue) > 0 {
		r, c := queue[0][0], queue[0][1]
		queue = queue[1:]
		res = grid[r][c]
		for _, d := range directions {
			nr, nc := r+d[0], c+d[1]
			if min(nr, nc) >= 0 && max(nr, nc) < n && grid[nr][nc] == 0 {
				queue = append(queue, [2]int{nr, nc})
				grid[nr][nc] = grid[r][c] + 1
			}
		}
	}
	if res > 1 {
		return res
	}
	return -1
}

// LC 1129
func shortestAlternatingPaths(n int, redEdges [][]int, blueEdges [][]int) []int {
	blue := map[int][]int{}
	red := map[int][]int{}
	for _, v := range redEdges {
		src, dst := v[0], v[1]
		red[src] = append(red[src], dst)
	}
	for _, v := range blueEdges {
		src, dst := v[0], v[1]
		blue[src] = append(blue[src], dst)
	}
	res := make([]int, n)
	for i := 0; i < n; i++ {
		res[i] = -1
	}
	queue := [][3]int{{0, 0, 0}}  // [node, dist, color] color=1 red, color=2 blue
	visited := make([][3]bool, n) // [node, color]
	visited[0][0] = true
	for len(queue) > 0 {
		node, dist, color := queue[0][0], queue[0][1], queue[0][2]
		queue = queue[1:]
		if res[node] == -1 {
			res[node] = dist
		}
		if color != 1 {
			for _, neighbor := range red[node] {
				if !visited[neighbor][1] {
					visited[neighbor][1] = true
					queue = append(queue, [3]int{neighbor, dist + 1, 1})
				}
			}
		}
		if color != 2 {
			for _, neighbor := range blue[node] {
				if !visited[neighbor][2] {
					visited[neighbor][2] = true
					queue = append(queue, [3]int{neighbor, dist + 1, 2})
				}
			}
		}
	}
	return res
}

// LC 2477
func minimumFuelCost(roads [][]int, seats int) int64 {
	adj := map[int][]int{}
	for _, v := range roads {
		src, dst := v[0], v[1]
		adj[src] = append(adj[src], dst)
		adj[dst] = append(adj[dst], src)
	}
	res := 0
	var dfs func(n, parent int) int
	dfs = func(n, parent int) int {
		passengers := 0
		for _, child := range adj[n] {
			if child != parent {
				p := dfs(child, n)
				passengers += p
				res += int(math.Ceil(float64(p) / float64(seats)))
			}
		}

		return passengers + 1 // +1 to include itself
	}
	dfs(0, 0)
	return int64(res)
}

// LC 2492
func minScoreTwoCities(n int, roads [][]int) int {
	adj := map[int][][2]int{}
	for _, v := range roads {
		src, dst, dist := v[0], v[1], v[2]
		adj[src] = append(adj[src], [2]int{dst, dist})
		adj[dst] = append(adj[dst], [2]int{src, dist})
	}
	res := math.MaxInt32
	visited := map[int]bool{}
	var dfs func(city int)
	dfs = func(city int) {
		if visited[city] {
			return
		}
		visited[city] = true
		for _, v := range adj[city] {
			nei, dist := v[0], v[1]
			res = min(res, dist)
			dfs(nei)
		}
	}
	dfs(1)
	return res
}

// LC 1254
func closedIsland(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	visited := make([][]bool, m)
	directions := [4][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}

	for i := range visited {
		visited[i] = make([]bool, n)
	}
	var dfs func(r, c int) int
	dfs = func(r, c int) int {
		if min(r, c) < 0 || r == m || c == n {
			return 0
		}
		if grid[r][c] == 1 || visited[r][c] {
			return 1
		}
		visited[r][c] = true
		isClosedLand := 1
		for _, d := range directions {
			nr, nc := r+d[0], c+d[1]
			isClosedLand = min(isClosedLand, dfs(nr, nc))
		}
		return isClosedLand
	}
	res := 0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] == 0 && !visited[i][j] {
				res += dfs(i, j)
			}
		}
	}
	return res
}

// LC 1020
func numEnclaves(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	directions := [4][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	visited := make([][]bool, m)
	for i := range visited {
		visited[i] = make([]bool, n)
	}
	var dfs func(r, c int) int
	dfs = func(r, c int) int {
		if min(r, c) < 0 || r == m || c == n || visited[r][c] || grid[r][c] == 0 {
			return 0
		}
		visited[r][c] = true
		res := 1
		for _, d := range directions {
			nr, nc := r+d[0], c+d[1]
			res += dfs(nr, nc)
		}
		return res
	}
	land, borderLand := 0, 0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			land += grid[i][j]
			if grid[i][j] == 1 && !visited[i][j] && (i == 0 || i == m-1 || j == 0 || j == n-1) {
				borderLand += dfs(i, j)
			}
		}
	}
	return land - borderLand
}

// LC 785
func isBipartite(graph [][]int) bool {
	odd := make([]int, len(graph))
	bfs := func(n int) bool {
		if odd[n] != 0 {
			return true
		}
		queue := []int{n}
		odd[n] = -1
		for len(queue) > 0 {
			n = queue[0]
			queue = queue[1:]
			for _, nei := range graph[n] {
				if odd[nei] != 0 && odd[n] == odd[nei] {
					return false
				} else if odd[nei] == 0 {
					queue = append(queue, nei)
					odd[nei] = -1 * odd[n]
				}
			}
		}
		return true
	}
	for i := range graph {
		if !bfs(i) {
			return false
		}
	}
	return true
}

// LC 399
func calcEquation(equations [][]string, values []float64, queries [][]string) []float64 {
	type denominator struct {
		deno  string
		value float64
	}
	adj := map[string][]denominator{}
	for i, v := range equations {
		a, b := v[0], v[1]
		adj[a] = append(adj[a], denominator{b, values[i]})
		adj[b] = append(adj[b], denominator{a, 1.0 / values[i]})
	}
	bfs := func(src, target string) float64 {
		if len(adj[src]) == 0 || len(adj[target]) == 0 {
			return -1
		}
		queue := []denominator{{src, 1}}
		visited := map[string]bool{src: true}
		for len(queue) > 0 {
			t := queue[0]
			queue = queue[1:]
			if t.deno == target {
				return t.value
			}
			for _, v := range adj[t.deno] {
				if !visited[v.deno] {
					visited[v.deno] = true
					queue = append(queue, denominator{v.deno, v.value * t.value})
				}
			}
		}

		return -1
	}
	res := make([]float64, len(queries))
	for i, v := range queries {
		res[i] = bfs(v[0], v[1])
	}
	return res
}

// LC 2101
func maximumDetonation(bombs [][]int) int {
	n := len(bombs)
	adj := make([][]int, n)
	for i := range bombs {
		for j := i + 1; j < n; j++ {
			x1, y1, r1 := bombs[i][0], bombs[i][1], bombs[i][2]
			x2, y2, r2 := bombs[j][0], bombs[j][1], bombs[j][2]
			d := (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)
			if d <= r1*r1 {
				adj[i] = append(adj[i], j)
			}
			if d <= r2*r2 {
				adj[j] = append(adj[j], i)
			}
		}
	}
	var dfs func(bomb int, visited []bool) int
	dfs = func(bomb int, visited []bool) int {
		visited[bomb] = true
		count := 1
		for _, b := range adj[bomb] {
			if !visited[b] {
				count += dfs(b, visited)
			}
		}
		return count
	}
	res := 0
	for i := range bombs {
		visited := make([]bool, n)
		res = max(res, dfs(i, visited))
	}
	return res
}

// LC 310
func findMinHeightTrees(n int, edges [][]int) []int {
	adj := make([][]int, n)
	indegree := make([]int, n)
	for _, e := range edges {
		a, b := e[0], e[1]
		adj[a] = append(adj[a], b)
		adj[b] = append(adj[b], a)
		indegree[a]++
		indegree[b]++
	}
	queue := []int{}
	for node := range indegree {
		if indegree[node] == 1 {
			queue = append(queue, node)
		}
	}
	// visit all leaf nodes
	for len(queue) > 0 {
		if n <= 2 {
			return queue
		}
		for _, node := range queue {
			queue = queue[1:]
			n--
			for _, nei := range adj[node] {
				indegree[nei]--
				if indegree[nei] == 1 {
					queue = append(queue, nei)
				}
			}
		}
	}
	return queue
}

// LC 1958
func checkMove(board [][]byte, rMove int, cMove int, color byte) bool {
	n := len(board)
	directions := [][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}}
	board[rMove][cMove] = color
	isLegalMove := func(r, c int, direction [2]int, color byte) bool {
		r, c = r+direction[0], c+direction[1]
		length := 1
		for 0 <= r && r < n && 0 <= c && c < n {
			length++
			if board[r][c] == '.' {
				return false
			}
			if board[r][c] == color {
				return length >= 3
			}
			r, c = r+direction[0], c+direction[1]
		}
		return false
	}
	for _, d := range directions {
		if isLegalMove(rMove, cMove, d, color) {
			return true
		}
	}
	return false
}

// 2359
func closestMeetingNode(edges []int, node1 int, node2 int) int {
	dist1, dist2 := make([]int, len(edges)), make([]int, len(edges))
	for i := 0; i < len(edges); i++ {
		dist1[i] = math.MaxInt32
		dist2[i] = math.MaxInt32
	}
	bfs := func(node int, dist []int) {
		queue := [][2]int{{node, 0}}
		dist[node] = 0
		for len(queue) > 0 {
			n, dst := queue[0][0], queue[0][1]
			queue = queue[1:]
			neighbor := edges[n]
			if neighbor != -1 && dist[neighbor] == math.MaxInt32 {
				queue = append(queue, [2]int{neighbor, dst + 1})
				dist[neighbor] = dst + 1
			}
		}
	}
	bfs(node1, dist1)
	bfs(node2, dist2)
	minNode, minDist := -1, math.MaxInt32
	for i := 0; i < len(edges); i++ {
		if minDist > max(dist1[i], dist2[i]) {
			minDist = max(dist1[i], dist2[i])
			minNode = i
		}
	}
	return minNode
}

// LC 1443
func minTimeToCollectApples(n int, edges [][]int, hasApple []bool) int {
	adj := make([][]int, n)
	for _, e := range edges {
		src, dst := e[0], e[1]
		adj[src] = append(adj[src], dst)
		adj[dst] = append(adj[dst], src)
	}
	var dfs func(node int, parent int) int
	dfs = func(node, parent int) int {
		time := 0
		for _, child := range adj[node] {
			if child == parent {
				continue
			}
			childTime := dfs(child, node)
			if childTime > 0 || hasApple[child] {
				time += 2 + childTime
			}
		}
		return time
	}
	return dfs(0, -1)
}
