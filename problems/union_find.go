package problems

import (
	"slices"
)

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

//	if n != u.parents[n] {
//	    u.parents[n] = u.Find(u.parents[n])
//	}
//
// return u.parents[n]
func (u *UnionFind) Find(n int) int {
	root := n
	for root != u.parents[root] {
		root = u.parents[root]
	}
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

// LC 785
func isBipartite(graph [][]int) bool {
	// odd := make([]int, len(graph))
	// bfs := func(n int) bool {
	// 	if odd[n] != 0 {
	// 		return true
	// 	}
	// 	queue := []int{n}
	// 	odd[n] = -1
	// 	for len(queue) > 0 {
	// 		n = queue[0]
	// 		queue = queue[1:]
	// 		for _, nei := range graph[n] {
	// 			if odd[nei] != 0 && odd[n] == odd[nei] {
	// 				return false
	// 			} else if odd[nei] == 0 {
	// 				queue = append(queue, nei)
	// 				odd[nei] = -1 * odd[n]
	// 			}
	// 		}
	// 	}
	// 	return true
	// }
	// for i := range graph {
	// 	if !bfs(i) {
	// 		return false
	// 	}
	// }
	// return true
	n := len(graph)
	u := NewUnionFind(n)
	for node := 0; node < n; node++ {
		for _, nei := range graph[node] {
			if u.Find(node) == u.Find(nei) {
				return false
			}
			u.Union(graph[node][0], nei)
		}
	}
	return true
}

// LC 886
func PossibleBipartition(n int, dislikes [][]int) bool {
	adj := make([][]int, n+1)
	for _, d := range dislikes {
		a, b := d[0], d[1]
		adj[a] = append(adj[a], b)
		adj[b] = append(adj[b], a)
	}
	u := NewUnionFind(n + 1)
	for node := 1; node <= n; node++ {
		for _, nei := range adj[node] {
			if u.Find(node) == u.Find(nei) {
				return false
			}
			u.Union(adj[node][0], nei)
		}
	}
	return true
}

// 1971
func validPath(n int, edges [][]int, source int, destination int) bool {
	u := NewUnionFind(n)
	for _, e := range edges {
		u.Union(e[0], e[1])
	}
	return u.Find(source) == u.Find(destination)
}

// LC 2421
func numberOfGoodPaths(vals []int, edges [][]int) int {
	adj := make([][]int, len(edges))
	for _, e := range edges {
		adj[e[0]] = append(adj[e[0]], e[1])
		adj[e[1]] = append(adj[e[1]], e[0])
	}
	valToIndex := map[int][]int{}
	for i, v := range vals {
		valToIndex[v] = append(valToIndex[v], i)
	}
	keys := make([]int, 0, len(valToIndex))
	for k := range valToIndex {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	u := NewUnionFind(len(vals))
	res := 0
	for _, v := range keys {
		for _, i := range valToIndex[v] {
			for _, nei := range adj[i] {
				if vals[nei] <= vals[i] {
					u.Union(nei, i)
				}
			}
		}
		count := map[int]int{}
		for _, i := range valToIndex[v] {
			count[u.Find(i)]++
			res += count[u.Find(i)] // 1+2+3
		}
	}
	return res
}

// LC 1579
func maxNumEdgesToRemove(n int, edges [][]int) int {
	alice, bob := NewUnionFind(n), NewUnionFind(n)
	count := 0
	for _, e := range edges {
		t, src, dst := e[0], e[1]-1, e[2]-1
		if t == 3 {
			al := alice.Union(src, dst)
			if bob.Union(src, dst) || al {
				count++
			}
		}
	}
	for _, e := range edges {
		t, src, dst := e[0], e[1]-1, e[2]-1
		if (t == 1 && alice.Union(src, dst)) || (t == 2 && bob.Union(src, dst)) {
			count++
		}
	}
	if alice.numsOfComponent == 1 && bob.numsOfComponent == 1 {
		return len(edges) - count
	}
	return -1
}

// LC 2709
func canTraverseAllPairs(nums []int) bool {
	u := NewUnionFind(len(nums))
	factorIndex := make(map[int]int)

	for i, n := range nums {
		j := 2
		for j*j <= n {
			if n%j == 0 {
				if f, ok := factorIndex[j]; ok {
					u.Union(i, f)
				} else {
					factorIndex[j] = i
				}
				for n%j == 0 {
					n = n / j
				}
			}
			j++
		}
		if n > 1 {
			if f, ok := factorIndex[n]; ok {
				u.Union(i, f)
			} else {
				factorIndex[n] = i
			}
		}
	}
	return u.numsOfComponent == 1
}
