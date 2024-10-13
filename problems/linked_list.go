package problems

// LC 21
func MergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	head := &ListNode{}
	cur := head
	for list1 != nil && list2 != nil {
		if list1.val <= list2.val {
			cur.next = &ListNode{val: list1.val}
			list1 = list1.next
		} else {
			cur.next = &ListNode{val: list2.val}
			list2 = list2.next
		}
		cur = cur.next
	}
	if list1 != nil {
		cur.next = list1
	} else {
		cur.next = list2
	}
	return head.next
}

// LC 23
func MergeKLists(lists []*ListNode) *ListNode {
	if len(lists) == 0 {
		return nil
	}
	for len(lists) > 1 {
		mergedList := []*ListNode{}
		for i := 0; i < len(lists); i += 2 {
			var l1 *ListNode = lists[i]
			var l2 *ListNode
			if i+1 < len(lists) {
				l2 = lists[i+1]
			}
			mergedList = append(mergedList, MergeTwoLists(l1, l2))
		}
		lists = mergedList
	}
	return lists[0]
}

// LC 25
func reverseKGroup(head *ListNode, k int) *ListNode {
	getKNode := func(h *ListNode, kth int) *ListNode {
		for h != nil && kth > 0 {
			h = h.next
			kth--
		}
		return h
	}
	dummy := &ListNode{next: head}
	prevGroup := dummy
	for {
		kNode := getKNode(prevGroup, k)
		if kNode == nil {
			break
		}
		nextGroup := kNode.next
		// reverse K group
		// 1->2->3->4|2->1->3->4 update new head as [prevGroup->1|prevGroup->2]
		prev, cur := nextGroup, prevGroup.next
		for cur != nextGroup {
			tmp := cur.next
			cur.next = prev
			prev = cur
			cur = tmp
		}
		tmp := prevGroup.next
		prevGroup.next = kNode
		prevGroup = tmp
	}
	return dummy.next
}

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

// LC 2095
func deleteMiddle(head *ListNode) *ListNode {
	dummy := &ListNode{0, 0, head}
	slow, fast := dummy, head
	for fast != nil && fast.next != nil {
		slow = slow.next
		fast = fast.next.next
	}
	slow.next = slow.next.next
	return dummy.next
}

// LC 146
type (
	DoublyListNode struct {
		key  int
		val  int
		next *DoublyListNode
		prev *DoublyListNode
	}
	LRUCache struct {
		cap   int
		cache map[int]*DoublyListNode
		left  *DoublyListNode // LRU
		right *DoublyListNode // MRU
	}
)

func NewLRUCache(capacity int) LRUCache {
	left := &DoublyListNode{}
	right := &DoublyListNode{}
	left.next = right
	right.prev = left
	return LRUCache{
		cap:   capacity,
		cache: make(map[int]*DoublyListNode),
		left:  left,
		right: right,
	}
}

func (this *LRUCache) remove(node *DoublyListNode) {
	prev, next := node.prev, node.next
	prev.next = next
	next.prev = prev
}

// insert at right most position
func (this *LRUCache) insert(node *DoublyListNode) {
	prev, next := this.right.prev, this.right
	prev.next = node
	node.next = next
	node.prev = prev
	next.prev = node
}

func (this *LRUCache) Get(key int) int {
	node := this.cache[key]
	if node != nil {
		this.remove(node)
		this.insert(node)
		return node.val
	}
	return -1
}

func (this *LRUCache) Put(key int, value int) {
	node := this.cache[key]
	// node existing we remove it
	if node != nil {
		this.remove(node)
	}
	node = &DoublyListNode{key: key, val: value}
	this.cache[key] = node
	this.insert(node)
	if len(this.cache) > this.cap {
		lru := this.left.next
		this.remove(lru)
		delete(this.cache, lru.key)
	}
}

// LC 460
type (
	LFUCache struct {
		capacity       int
		countToLRUKeys map[int]*DoublyLinkedList // freq -> doubly LinkedList
		keyToVal       map[int]int
		keyToCount     map[int]int
		leastFreq      int
	}
	DoublyLinkedList struct {
		head      *DoublyListNode
		tail      *DoublyListNode
		keyToNode map[int]*DoublyListNode
	}
)

func NewDoublyLinkedList() *DoublyLinkedList {
	head := &DoublyListNode{}
	tail := &DoublyListNode{}
	head.next = tail
	tail.prev = head
	return &DoublyLinkedList{
		head,
		tail,
		map[int]*DoublyListNode{},
	}
}

func (this *DoublyLinkedList) Pop(key int) {
	node := this.keyToNode[key]
	if node != nil {
		next, prev := node.next, node.prev
		prev.next = next
		next.prev = prev
		delete(this.keyToNode, key)
	}
}

func (this *DoublyLinkedList) PopLeft() int {
	node := this.head.next
	this.Pop(node.key)
	return node.key
}

func (this *DoublyLinkedList) Push(key int) {
	node := &DoublyListNode{key: key, prev: this.tail.prev, next: this.tail}
	this.tail.prev = node
	this.keyToNode[key] = node
	node.prev.next = node
}

func (this *DoublyLinkedList) Length() int {
	return len(this.keyToNode)
}

func NewLFUCache(capacity int) LFUCache {
	return LFUCache{
		capacity:       capacity,
		leastFreq:      1,
		keyToVal:       map[int]int{},
		keyToCount:     map[int]int{},
		countToLRUKeys: map[int]*DoublyLinkedList{},
	}
}

func (this *LFUCache) updateCounter(key int) {
	count := this.keyToCount[key]
	this.keyToCount[key]++
	if this.countToLRUKeys[count] == nil {
		this.countToLRUKeys[count] = NewDoublyLinkedList()
	}
	if this.countToLRUKeys[count+1] == nil {
		this.countToLRUKeys[count+1] = NewDoublyLinkedList()
	}
	this.countToLRUKeys[count].Pop(key)
	this.countToLRUKeys[count+1].Push(key)
	if count == this.leastFreq && this.countToLRUKeys[count].Length() == 0 {
		this.leastFreq++
	}
}

func (this *LFUCache) Get(key int) int {
	value, isKeyExist := this.keyToVal[key]
	if !isKeyExist {
		return -1
	}
	this.updateCounter(key)
	return value
}

func (this *LFUCache) Put(key int, value int) {
	if this.capacity == 0 {
		return
	}
	// evict LRU
	if _, ok := this.keyToVal[key]; !ok && len(this.keyToVal) == this.capacity {
		k := this.countToLRUKeys[this.leastFreq].PopLeft()
		delete(this.keyToVal, k)
		delete(this.keyToCount, k)
	}
	this.keyToVal[key] = value
	this.updateCounter(key)
	this.leastFreq = min(this.leastFreq, this.keyToCount[key])
}

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

// LC 234
func isPalindromeLinkedList(head *ListNode) bool {
	reverse := func(node *ListNode) *ListNode {
		var prev *ListNode
		var cur = node
		for cur != nil {
			next := cur.next
			cur.next, prev = prev, cur
			cur = next
		}
		return prev
	}
	slow, fast := head, head
	for fast != nil && fast.next != nil {
		slow = slow.next
		fast = fast.next.next
	}
	node := reverse(slow)
	for head != nil {
		if head.val != node.val {
			return false
		}
		head = head.next
		node = node.next
	}
	return true
}

// LC 2487 monotonic stack
func removeNodes(head *ListNode) *ListNode {
	node := head
	stack := []*ListNode{}
	for node != nil {
		for len(stack) > 0 && node.val > stack[len(stack)-1].val {
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, node)
		node = node.next
	}
	// head reversal
	head = nil
	for i := len(stack) - 1; i >= 0; i-- {
		stack[i].next = head
		head = stack[i]
	}
	return head
}

// LC 19
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := &ListNode{next: head}
	prev, cur := dummy, head
	for range n {
		cur = cur.next
	}
	for cur != nil {
		prev = prev.next
		cur = cur.next
	}
	prev.next = prev.next.next
	return dummy.next
}
