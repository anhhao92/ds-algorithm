package problems

import (
	"strings"
)

type TrieNode struct {
	children [26]*TrieNode
	isWord   bool
}

func FindWords(board [][]byte, words []string) []string {
	result := []string{}
	row, col := len(board), len(board[0])
	visited := make([][]bool, row)
	root := &TrieNode{}
	for i := 0; i < row; i++ {
		visited[i] = make([]bool, col)
	}
	for _, w := range words {
		curNode := root
		for i := 0; i < len(w); i++ {
			char := w[i] - 'a'
			if curNode.children[char] == nil {
				curNode.children[char] = &TrieNode{}
			}
			curNode = curNode.children[char]
		}
		curNode.isWord = true
	}
	var dfs func(r, c int, trie *TrieNode, word []byte)
	dfs = func(r, c int, trie *TrieNode, word []byte) {
		if r < 0 || c < 0 || r >= row || c >= col || visited[r][c] {
			return
		}
		visited[r][c] = true
		char := board[r][c] - 'a'
		curTrie := trie.children[char]
		if curTrie == nil {
			visited[r][c] = false
			return
		}
		word = append(word, char)
		if curTrie.isWord {
			result = append(result, string(word))
			curTrie.isWord = false
		}
		directions := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
		for _, d := range directions {
			dr, dc := r+d[0], c+d[1]
			dfs(dr, dc, curTrie, word)
		}
		visited[r][c] = false
	}
	// traverse cells
	for i := 0; i < row; i++ {
		for j := 0; j < col; j++ {
			dfs(i, j, root, []byte{})
		}
	}
	return result
}

/**
 * Your Trie object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Insert(word);
 * param_2 := obj.Search(word);
 * param_3 := obj.StartsWith(prefix);
 */

type Trie struct {
	children map[byte]*Trie
	isWord   bool
}

func TrieConstructor() Trie {
	return Trie{children: make(map[byte]*Trie)}
}

func (this *Trie) Insert(word string) {
	current := this
	for i := 0; i < len(word); i++ {
		if current.children[word[i]] == nil {
			current.children[word[i]] = &Trie{children: make(map[byte]*Trie)}
		}
		current = current.children[word[i]]
	}
	current.isWord = true
}

func (this *Trie) Search(word string) bool {
	current := this
	for i := 0; i < len(word); i++ {
		if current.children[word[i]] == nil {
			return false
		}
		current = current.children[word[i]]
	}
	return current.isWord
}

func (this *Trie) StartsWith(prefix string) bool {
	current := this
	for i := 0; i < len(prefix); i++ {
		if current.children[prefix[i]] == nil {
			return false
		}
		current = current.children[prefix[i]]
	}
	return true
}

/**
 * Your MapSum object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Insert(key,val);
 * param_2 := obj.Sum(prefix);
 */
type PrefixTrie struct {
	children [26]*PrefixTrie
	value    int
}

type MapSum struct {
	hashMap map[string]int
	root    *PrefixTrie
}

func Constructor() MapSum {
	return MapSum{
		hashMap: make(map[string]int),
		root:    &PrefixTrie{}}
}

func (this *MapSum) Insert(key string, val int) {
	current := this.root
	for i := 0; i < len(key); i++ {
		c := key[i] - 'a'
		if current.children[c] == nil {
			current.children[c] = &PrefixTrie{}
		}
		current = current.children[c]
		current.value += val
		if value, hasKey := this.hashMap[key]; hasKey {
			current.value -= value
		}
	}
	this.hashMap[key] = val
}

func (this *MapSum) Sum(prefix string) int {
	current := this.root
	for i := 0; i < len(prefix); i++ {
		c := prefix[i] - 'a'
		if current.children[c] == nil {
			return 0
		}
		current = current.children[c]
	}
	return current.value
}

// LC 2416
func SumPrefixScores(words []string) []int {
	root := &PrefixTrie{}
	for _, w := range words {
		current := root
		for i := 0; i < len(w); i++ {
			c := w[i] - 'a'
			if current.children[c] == nil {
				current.children[c] = &PrefixTrie{}
			}
			current = current.children[c]
			current.value++
		}
	}
	res := make([]int, len(words))
	for k, w := range words {
		score := 0
		current := root
		for i := 0; i < len(w); i++ {
			c := w[i] - 'a'
			current = current.children[c]
			score += current.value
		}
		res[k] = score
	}
	return res
}

// LC 648
func ReplaceWords(dictionary []string, sentence string) string {
	words := strings.Split(sentence, " ")
	root := &TrieNode{}
	for _, w := range dictionary {
		cur := root
		for i := 0; i < len(w); i++ {
			c := w[i] - 'a'
			if cur.children[c] == nil {
				cur.children[c] = &TrieNode{}
			}
			cur = cur.children[c]
		}
		cur.isWord = true
	}
	for i, w := range words {
		cur := root
		for j := 0; j < len(w); j++ {
			c := w[j] - 'a'
			if cur.children[c] == nil {
				break
			}
			cur = cur.children[c]
			if cur.isWord {
				words[i] = w[:j+1]
				break
			}
		}
	}
	return strings.Join(words, " ")
}
