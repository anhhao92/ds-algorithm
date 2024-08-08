package problems

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
	children map[byte]*PrefixTrie
	value    int
}

type MapSum struct {
	hashMap map[string]int
	root    *PrefixTrie
}

func Constructor() MapSum {
	return MapSum{
		hashMap: make(map[string]int),
		root:    &PrefixTrie{children: make(map[byte]*PrefixTrie)}}
}

func (this *MapSum) Insert(key string, val int) {
	current := this.root
	for i := 0; i < len(key); i++ {
		if current.children[key[i]] == nil {
			current.children[key[i]] = &PrefixTrie{children: make(map[byte]*PrefixTrie)}
		}
		current = current.children[key[i]]
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
		if current.children[prefix[i]] == nil {
			return 0
		}
		current = current.children[prefix[i]]
	}
	return current.value
}
