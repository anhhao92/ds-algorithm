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

// LC 143
func reorderList(head *ListNode) {
	slow, fast := head, head.next
	for fast != nil && fast.next != nil {
		slow = slow.next
		fast = fast.next.next
	}
	// reverse the half
	half := slow.next
	slow.next = nil
	prev := slow.next
	for half != nil {
		next := half.next
		half.next = prev
		prev = half
		half = next
	}

	first, second := head, prev
	for second != nil {
		next1, next2 := first.next, second.next
		first.next = second
		second.next = next1
		first, second = next1, next2
	}
}

// LC 86
func partitionList(head *ListNode, x int) *ListNode {
	left, right := &ListNode{}, &ListNode{}
	leftTail, rightTail := left, right
	for head != nil {
		if head.val < x {
			leftTail.next = head
			leftTail = leftTail.next
		} else {
			rightTail.next = head
			rightTail = rightTail.next
		}
		head = head.next
	}
	leftTail.next = right.next
	rightTail.next = nil
	return left.next
}

// LC 61
func rotateRight(head *ListNode, k int) *ListNode {
	if head == nil {
		return head
	}
	tail := head
	n := 1
	for tail.next != nil {
		tail = tail.next
		n++
	}
	k = k % n
	if k == 0 {
		return head
	}
	// move to n - k - 1
	cur := head
	for range n - k - 1 {
		cur = cur.next
	}
	newHead := cur.next
	cur.next = nil
	tail.next = head
	return newHead
}

// LC 148
func sortList(head *ListNode) *ListNode {
	if head == nil || head.next == nil {
		return head
	}
	findMid := func(n *ListNode) *ListNode {
		slow, fast := n, n.next
		for fast != nil && fast.next != nil {
			slow = slow.next
			fast = fast.next.next
		}
		return slow
	}
	merge := func(l, r *ListNode) *ListNode {
		h := &ListNode{}
		cur := h
		for l != nil && r != nil {
			if l.val <= r.val {
				cur.next = l
				l = l.next
			} else {
				cur.next = r
				r = r.next
			}
			cur = cur.next
		}
		if l != nil {
			cur.next = l
		}
		if r != nil {
			cur.next = r
		}
		return h.next
	}
	left := head
	right := findMid(head)
	// split 2 half
	temp := right.next
	right.next = nil
	right = temp

	left = sortList(left)
	right = sortList(right)
	return merge(left, right)
}

// LC 147
func insertionSortList(head *ListNode) *ListNode {
	dummy := &ListNode{next: head}
	cur, prev := head.next, head
	for cur != nil {
		if cur.val >= prev.val {
			prev, cur = cur, cur.next
			continue
		}
		// tmp -> 5 [6, prev] [4, cur] [1, next]
		// tmp -> [4, cur], 5 [6, prev] 1
		temp := dummy
		for cur.val > temp.next.val {
			temp = temp.next
		}
		prev.next = cur.next
		cur.next = temp.next
		temp.next = cur
		cur = prev.next
	}
	return dummy.next
}

// LC 1669
func mergeInBetween(list1 *ListNode, a int, b int, list2 *ListNode) *ListNode {
	cur := list1
	count := 0
	for count < a-1 {
		count++
		cur = cur.next
	}
	prev1 := cur
	for count <= b {
		count++
		cur = cur.next
	}
	prev1.next = list2
	for list2.next != nil {
		list2 = list2.next
	}
	list2.next = cur
	return list1
}

// LC 1721
func swapNodes(head *ListNode, k int) *ListNode {
	cur := head
	for range k - 1 {
		cur = cur.next
	}
	l, r := cur, head
	// 2 pointers shift to the end
	for cur.next != nil {
		cur = cur.next
		r = r.next
	}
	// l, r will be at k(th) position
	l.val, r.val = r.val, l.val
	return head
}

// LC 2130
func pairSum(head *ListNode) int {
	res := 0
	slow, fast := head, head
	var prev *ListNode
	// reverse half
	for fast != nil && fast.next != nil {
		fast = fast.next.next
		tmp := slow.next
		slow.next, prev = prev, slow
		slow = tmp
	}
	for slow != nil {
		res = max(res, prev.val+slow.val)
		slow = slow.next
		prev = prev.next
	}
	return res
}

// LC 24
func swapPairs(head *ListNode) *ListNode {
	dummy := &ListNode{next: head}
	prev, cur := dummy, head
	for cur != nil && cur.next != nil {
		nextPairs := cur.next.next
		second := cur.next
		// reverse pairs (second, cur)
		second.next = cur
		cur.next = nextPairs
		prev.next = second
		// update prev -> cur
		prev = cur
		cur = nextPairs
	}
	return dummy.next
}

// LC 725
func splitListToParts(head *ListNode, k int) []*ListNode {
	cur, n := head, 0
	for cur != nil {
		n++
		cur = cur.next
	}
	res := make([]*ListNode, k)
	remainder := n % k
	baseLen := n / k
	cur = head
	for i := 0; i < n && cur != nil; i++ {
		res[i] = cur
		length := baseLen
		if remainder > 0 {
			length++
			remainder--
		}
		for j := 0; j < length-1 && cur != nil; j++ {
			cur = cur.next
		}
		if cur != nil {
			tmp := cur.next
			cur.next = nil
			cur = tmp
		}
	}
	return res
}

// LC 1472
type (
	BrowserHistory struct {
		head *DoublyNode
	}
	DoublyNode struct {
		val  string
		prev *DoublyNode
		next *DoublyNode
	}
)

func NewBrowserHistory(homepage string) BrowserHistory {
	return BrowserHistory{&DoublyNode{val: homepage}}
}

func (this *BrowserHistory) Visit(url string) {
	this.head.next = &DoublyNode{val: url, prev: this.head}
	this.head = this.head.next
}

func (this *BrowserHistory) Back(steps int) string {
	for this.head.prev != nil && steps > 0 {
		this.head = this.head.prev
		steps--
	}
	return this.head.val
}

func (this *BrowserHistory) Forward(steps int) string {
	for this.head.next != nil && steps > 0 {
		this.head = this.head.next
		steps--
	}
	return this.head.val
}

// LC 707
type MyLinkedList struct {
	head *DoublyListNode
	tail *DoublyListNode
}

func NewMyLinkedList() MyLinkedList {
	h := &DoublyListNode{}
	t := &DoublyListNode{}
	h.next = t
	t.prev = h
	return MyLinkedList{h, t}
}

func (this *MyLinkedList) Get(index int) int {
	cur := this.head.next
	for cur != nil && index > 0 {
		cur = cur.next
		index--
	}
	if cur != nil && cur != this.tail && index == 0 {
		return cur.val
	}
	return -1
}

func (this *MyLinkedList) AddAtHead(val int) {
	node := &DoublyListNode{val: val}
	next, prev := this.head.next, this.head
	node.next = next
	node.prev = prev
	prev.next = node
	next.prev = node
}

func (this *MyLinkedList) AddAtTail(val int) {
	node := &DoublyListNode{val: val}
	next, prev := this.tail, this.tail.prev
	prev.next = node
	next.prev = node
	node.next = next
	node.prev = prev
}

func (this *MyLinkedList) AddAtIndex(index int, val int) {
	cur := this.head.next
	for cur != nil && index > 0 {
		cur = cur.next
		index--
	}
	if cur != nil && index == 0 {
		node := &DoublyListNode{val: val}
		next, prev := cur, cur.prev
		node.next = next
		node.prev = prev
		prev.next = node
		next.prev = node
	}
}

func (this *MyLinkedList) DeleteAtIndex(index int) {
	cur := this.head.next
	for cur != nil && index > 0 {
		cur = cur.next
		index--
	}
	if cur != nil && cur != this.tail && index == 0 {
		next, prev := cur.next, cur.prev
		prev.next = next
		next.prev = prev
	}
}
