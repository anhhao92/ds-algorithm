package problems

// LC 287 Floyd's cycle detection
func FindDuplicate(nums []int) int {
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

/**
 * Your MyHashMap object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Put(key,value);
 * param_2 := obj.Get(key);
 * obj.Remove(key);
 */
// LC705
type ListNode struct {
	key  int
	val  int
	next *ListNode
}
type MyHashMap struct {
	data []*ListNode
}

func NewMyHashMap() MyHashMap {
	// create dummy node as head node
	data := make([]*ListNode, 1024)
	for i := 0; i < len(data); i++ {
		data[i] = &ListNode{}
	}
	return MyHashMap{data}
}

func (this *MyHashMap) Hash(key int) int {
	return key % len(this.data)
}

func (this *MyHashMap) Put(key int, value int) {
	hashKey := this.Hash(key)
	node := this.data[hashKey]
	for node != nil && node.next != nil {
		// update current key
		if node.next.key == key {
			node.next.val = value
			return
		}
		node = node.next
	}
	node.next = &ListNode{key: key, val: value}
}

func (this *MyHashMap) Get(key int) int {
	hashKey := this.Hash(key)
	node := this.data[hashKey]
	for node != nil && node.next != nil {
		if node.next.key == key {
			return node.next.val
		}
		node = node.next
	}
	return -1
}

func (this *MyHashMap) Remove(key int) {
	hashKey := this.Hash(key)
	node := this.data[hashKey]
	for node != nil && node.next != nil {
		if node.next.key == key {
			node.next = node.next.next
			return
		}
		node = node.next
	}
}

// LC706
type MyHashSet struct {
	data []*ListNode
}

func NewMyHashSet() MyHashSet {
	// create dummy node as head node
	data := make([]*ListNode, 1024)
	for i := 0; i < len(data); i++ {
		data[i] = &ListNode{}
	}
	return MyHashSet{data}
}

func (this *MyHashSet) Hash(key int) int {
	return key % len(this.data)
}

func (this *MyHashSet) Add(key int) {
	hashKey := this.Hash(key)
	node := this.data[hashKey]
	for node != nil && node.next != nil {
		// current key existed
		if node.next.key == key {
			return
		}
		node = node.next
	}
	node.next = &ListNode{key: key}
}

func (this *MyHashSet) Remove(key int) {
	hashKey := this.Hash(key)
	node := this.data[hashKey]
	for node != nil && node.next != nil {
		if node.next.key == key {
			node.next = node.next.next
			return
		}
		node = node.next
	}
}

func (this *MyHashSet) Contains(key int) bool {
	hashKey := this.Hash(key)
	node := this.data[hashKey]
	for node != nil && node.next != nil {
		// current key existed
		if node.next.key == key {
			return true
		}
		node = node.next
	}
	return false
}

// LC 92
func ReverseLinkedListBetweenLeftRight(head *ListNode, left int, right int) *ListNode {
	dummy := &ListNode{next: head}
	leftPrev, cur := dummy, head
	for i := 0; i < left-1; i++ {
		leftPrev, cur = cur, cur.next
	}
	var prev *ListNode
	for i := 0; i < right-left+1; i++ {
		temp := cur.next
		cur.next = prev
		prev = cur
		cur = temp
	}
	leftPrev.next.next = cur
	leftPrev.next = prev
	return dummy.next
}
