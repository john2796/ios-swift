# ----------------- Arrays & Hashing -----------------
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = {}  # {1:3, 2:2, 3:1}
        freq = [[] for i in range(len(nums) + 1)]  # [[], [], [], []]
        for n in nums:
            count[n] = 1 + count.get(n, 0)
        for key, value in count.items():
            freq[value].append(key)

        # print(freq) #  [[], [3], [2], [1], [], [], []]
        res = []
        for i in range(
            len(freq) - 1, 0, -1
        ):  # # loop through freq in reverse, then store values in res, once k equals res return result.
            for n in freq[i]:
                res.append(n)
                if k == len(res):
                    return res


class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res = [1] * (len(nums))

        for i in range(1, len(nums)):
            res[i] = res[i - 1] * nums[i - 1]
        postfix = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= postfix
            postfix *= nums[i]
        return res


"""
128. Longest Consecutive Sequence
return the length of the longest conseccutive elments sequence.
Example 1:

Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.
"""


class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        numSet = set(nums)
        longest = 0

        for n in numSet:
            # check if start of sequence
            if (n - 1) not in numSet:
                length = 1
                while (n + length) in numSet:
                    length += 1
                longest = max(longest, length)
        return longest


# ----------------- Two Pointers -----------------
"""
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

 

Example 1:

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Explanation: 
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
The distinct triplets are [-1,0,1] and [-1,-1,2].
Notice that the order of the output and the order of the triplets does not matter.

Approach:
1. sort the nums
2. iterate nums, edge cases greater than zero skip and index greater than zero and current and previous arent the same 
3. do regular two pointer
4. shrink window otherwise add the triplets
"""


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # return all triplets that equals to zero, must not contain duplicate triplets
        res = []
        nums.sort()

        for i, a in enumerate(nums):
            # skip positive integers
            if a > 0:
                break
            if i > 0 and a == nums[i - 1]:
                continue
            l, r = i + 1, len(nums) - 1
            while l < r:
                threeSum = a + nums[l] + nums[r]
                if threeSum > 0:  # shrink right side
                    r -= 1
                elif threeSum < 0:  # shrink left side
                    l += 1
                else:  # return result
                    res.append([a, nums[l], nums[r]])
                    l += 1  # shrink window
                    r -= 1
                    while (
                        nums[l] == nums[l - 1] and l < r
                    ):  # move left pointer when l and prev are the same and ensure it doesn't go out of bounds
                        l += 1
        return res


class Solution:
    def maxArea(self, height: List[int]) -> int:
        # return the maximum amount of water a container can store
        # [1,8,6,2,5,4,8,3,7] , 7 * 7 = 49
        # if left is less than right the water will overflow
        l, r = 0, len(height) - 1
        res = 0

        while l < r:
            area = min(height[r], height[l]) * (r - l)
            res = max(res, area)
            if height[l] < height[r]:
                l += 1
            elif height[r] <= height[l]:
                r -= 1
        return res


# ----------------- Sliding Window -----------------
"""
Input: prices = [7,1,5,3,6,4]
profit = buy low on day one sell high future day
buy = 1, sell =6 , profit = 5

Output: 5
"""


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # return the maximum profit you can achieve
        res = 0
        lowest = prices[0]

        for price in prices:
            if price < lowest:
                lowest = price
            res = max(res, price - lowest)
        return res


"""
 find the length of the longest substring without repeating characters.
 sliding window + hashset
Example 1:

Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
"""


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        hashset = set()
        l = 0
        res = 0

        for r in range(len(s)):
            while s[r] in hashset:
                hashset.remove(s[l])
                l += 1
            hashset.add(s[r])
            res = max(res, r - l + 1)
        return res


class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        # change character to uppercase k times, i wanna know the longest substring containing same letter
        # "ABAB", k=2 , output=4 , we change A or B k times(2) we get 4 substring

        # sliding window +  char count
        l, maxf = 0, 0
        count = {}

        for r in range(len(s)):
            # expand window + calc maxf
            count[s[r]] = 1 + count.get(s[r], 0)
            maxf = max(maxf, count[s[r]])

            # shrink window
            if (r - l + 1) - maxf > k:
                count[s[l]] -= 1
                l += 1
        return r - l + 1  # how did i get access to r from here??


"""
Example 1:

Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
# sliding window + hashset
"""


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if t == "":
            return ""
        countT, window = {}, {}
        # store t count in hashset
        for c in t:
            countT[c] = 1 + countT.get(c, 0)

        have, need = 0, len(countT)
        res, resLen = [-1, -1], float("infinity")

        l = 0
        for r in range(len(s)):
            # expand window
            c = s[r]
            window[c] = 1 + window.get(c, 0)

            if c in countT and window[c] == countT[c]:
                have += 1
            # shrink window
            while have == need:
                # update our result
                if (r - l + 1) < resLen:
                    res = [l, r]
                    resLen = r - l + 1
                # pop from left of our window
                window[s[l]] -= 1
                if s[l] in countT and window[s[l]] < countT[s[l]]:
                    have -= 1
                l += 1
        l, r = res
        return s[l : r + 1] if resLen != float("infinity") else ""


# ----------------- Stack -----------------


# store pair in hashKey, use stack to check if matches hashKey
# Stack follows LIFO while Queue follows FIFO data structure type.
class Solution:
    def isValid(self, s: str) -> bool:
        # [( )]
        hashKey = {")": "(", "}": "{", "]": "["}
        stack = []
        for c in s:
            if c not in hashKey:
                stack.append(c)
                continue
            if not stack or hashKey[c] != stack[-1]:
                return False
            stack.pop()
        return not stack


# ----------------- Binary Search -----------------
"""
Given the sorted rotated array nums of unique elements, return the minimum element of this array.

Example 1:

Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.
"""


class Solution:
    def findMin(self, nums: List[int]) -> int:
        start, end = 0, len(nums) - 1
        curr_min = float("inf")

        while start < end:
            mid = start + (end - start) // 2  # pivot
            curr_min = min(curr_min, nums[mid])  # store current min

            # decide where to place indices
            # right has the min
            if nums[mid] > nums[end]:
                start = mid + 1
            # left has the min
            else:
                end = mid - 1
        return min(curr_min, nums[start])


"""
Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

"""


# check wether to do binary search on left sorted or right sorted move pointer based on pivot middle less than or greater than
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1

        while l <= r:
            mid = (l + r) // 2
            if target == nums[mid]:
                return mid
            # left sorted position
            if nums[l] <= nums[mid]:
                if target > nums[mid] or target < nums[l]:
                    l = mid + 1
                else:
                    r = mid - 1
            # right sorted position
            else:
                if target < nums[mid] or target > nums[r]:
                    r = mid - 1
                else:
                    l = mid + 1
        return -1


# ----------------- Linked List -----------------
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr = None, head

        while curr:
            temp = curr.next  # store original curr
            curr.next = prev  # set curr next to prev
            prev = curr  # prev to next swap
            curr = temp  # grab original temp set to curr
        return prev


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(
        self, list1: Optional[ListNode], list2: Optional[ListNode]
    ) -> Optional[ListNode]:
        dummy = node = ListNode()

        while list1 and list2:
            if list1.val < list2.val:
                node.next = list1
                list1 = list1.next
            else:
                node.next = list2
                list2 = list2.next
            node = node.next
        node.next = list1 or list2
        return dummy.next


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        # find middle
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        # print(slow.val , fast.val) # slow == 2, fast ==4

        # reverse second half
        second = slow.next
        prev = slow.next = None
        # print(second.val, prev) # 3 None
        while second:
            tmp = second.next
            second.next = prev
            prev = second
            second = tmp
        # print(prev.val) # 4

        # merge two halfs
        first, second = head, prev  # 1, 4
        while second:
            tmp1, tmp2 = first.next, second.next
            first.next = second
            second.next = tmp1
            first, second = tmp1, tmp2


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        # starting from tail remove nth node
        dummy = ListNode(0, head)
        left = dummy
        right = head

        while n > 0:
            right = right.next
            n -= 1
        while right:
            left = left.next
            right = right.next
        # delete
        left.next = left.next.next
        return dummy.next


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        # check for empty lists
        if not lists or len(lists) == 0:
            return None

        while len(lists) > 1:
            mergedLists = []
            # loop list get l1 and l2
            for i in range(0, len(lists), 2):
                l1 = lists[i]
                l2 = lists[i + 1] if (i + 1) < len(lists) else None
                mergedLists.append(self.mergeList(l1, l2))
            lists = mergedLists
        return lists[0]

    # merge two linkedlists
    def mergeList(self, l1, l2):
        dummy = ListNode()
        tail = dummy

        while l1 and l2:
            if l1.val < l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
        if l1:
            tail.next = l1
        if l2:
            tail.next = l2
        return dummy.next


# ----------------- Trees -----------------
# Recursive DFS
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        left = 1 + self.maxDepth(root.left)
        right = 1 + self.maxDepth(root.right)

        return max(left, right)


# Iterative DFS
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        stack = [[root, 1]]
        res = 0
        while stack:
            node, depth = stack.pop()
            if node:
                res = max(res, depth)
                stack.append([node.left, depth + 1])
                stack.append([node.right, depth + 1])
        return res


# Iterative BFS
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        q = deque()
        if root:
            q.append(root)
        level = 0

        while q:
            for i in range(len(q)):
                node = q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            level += 1
        return level


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if p and q and p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        else:
            return False


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, s: Optional[TreeNode], t: Optional[TreeNode]) -> bool:
        if not t:
            return True
        if not s:
            return False
        if self.sameTree(s, t):
            return True
        return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)

    def sameTree(self, s, t):
        if not s and not t:
            return True
        if s and t and s.val == t.val:
            return self.sameTree(s.left, t.left) and self.sameTree(s.right, t.right)
        return False


"""
Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
Output: 6
Explanation: The LCA of nodes 2 and 8 is 6.
"""


class Solution:
    def lowestCommonAncestor(
        self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":
        while True:
            if root.val < p.val and root.val < q.val:
                root = root.right
            elif root.val > p.val and root.val > q.val:
                root = root.left
            else:
                return root


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        q = deque()
        if root:
            q.append(root)
        res = []

        while q:
            val = []
            for i in range(len(q)):
                node = q.popleft()
                val.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(val)
        return res


class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        # left is less than root
        # right is greater than root
        def valid(node, left, right):
            if not node:
                return True
            if not (left < node.val < right):
                return False
            return valid(node.left, left, node.val) and valid(
                node.right, node.val, right
            )

        return valid(root, float("-inf"), float("inf"))


class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        # dfs iterative
        stack = []
        curr = root
        while stack or curr:
            while curr:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            k -= 1
            if k == 0:
                return curr.val
            curr = curr.right


"""
Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.

Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]
"""


class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        # combine both preorder and inorder traversal
        if not preorder or not inorder:
            return None
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1 : mid + 1], inorder[:mid])
        root.right = self.buildTree(preorder[mid + 1 :], inorder[mid + 1 :])
        return root


"""
A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.

The path sum of a path is the sum of the node's values in the path.

Given the root of a binary tree, return the maximum path sum of any non-empty path.

 

Example 1:


Input: root = [1,2,3]
Output: 6
Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.

"""


class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        res = [root.val]

        # return max path sum without split
        def dfs(root):
            if not root:
                return 0

            leftMax = dfs(root.left)
            rightMax = dfs(root.right)
            leftMax = max(leftMax, 0)
            rightMax = max(rightMax, 0)

            # compute max path sum WITH split
            res[0] = max(res[0], root.val + leftMax + rightMax)
            return root.val + max(leftMax, rightMax)

        dfs(root)
        return res[0]


"""
Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

Clarification: The input/output format is the same as how LeetCode serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.

Example 1:


Input: root = [1,2,3,null,null,4,5]
Output: [1,2,3,null,null,4,5]

"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        res = []

        def dfs(node):
            if not node:
                res.append("N")
                return
            res.append(str(node.val))
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return ",".join(res)

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        vals = data.split(",")
        self.i = 0

        def dfs():
            if vals[self.i] == "N":
                self.i += 1
                return None
            node = TreeNode(int(vals[self.i]))
            self.i += 1
            node.left = dfs()
            node.right = dfs()
            return node

        return dfs()


# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))


# ----------------- Tries -----------------
class TrieNode:
    def __init__(self):
        self.children = [None] * 26
        self.end = False


"""
Implement the Trie class:

- Trie() Initializes the trie object.
- void insert(String word) Inserts the string word into the trie.
- boolean search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
- boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.
"""


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        # insert a word into trie
        curr = self.root
        for c in word:
            # index in alphabet
            i = ord(c) - ord("a")
            # check if curr children is None add TrieNode
            if curr.children[i] == None:
                curr.children[i] = TrieNode()
            curr = curr.children[i]
        curr.end = True  # mark last character

    def search(self, word: str) -> bool:
        curr = self.root
        for c in word:
            i = ord(c) - ord("a")
            if curr.children[i] == None:
                return False
            curr = curr.children[i]
        return curr.end

    def startsWith(self, prefix: str) -> bool:
        curr = self.root
        for p in prefix:
            i = ord(p) - ord("a")
            if curr.children[i] == None:
                return False
            curr = curr.children[i]
        return True


"""
Design a data structure that supports adding new words and finding if a string matches any previously added string.

Implement the WordDictionary class:

WordDictionary() Initializes the object.
void addWord(word) Adds word to the data structure, it can be matched later.
bool search(word) Returns true if there is any string in the data structure that matches word or false otherwise. word may contain dots '.' where dots can be matched with any letter.
 

Example:

Input
["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
[[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
Output
[null,null,null,null,false,true,true,true]

Explanation
WordDictionary wordDictionary = new WordDictionary();
wordDictionary.addWord("bad");
wordDictionary.addWord("dad");
wordDictionary.addWord("mad");
wordDictionary.search("pad"); // return False
wordDictionary.search("bad"); // return True
wordDictionary.search(".ad"); // return True
wordDictionary.search("b.."); // return True
"""


# same idea as Trie Class edge case it may contain '.' where dots can be mathed with any letter
# use object instead of array of empty 26 alphabet
class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = False


class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.word = True

    def search(self, word: str) -> bool:
        def dfs(j, root):
            cur = root
            for i in range(j, len(word)):
                c = word[i]
                if c == ".":
                    for child in cur.children.values():
                        if dfs(i + 1, child):
                            return True
                    return False
                else:
                    if c not in cur.children:
                        return False
                    cur = cur.children[c]
            return cur.word

        return dfs(0, self.root)


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)


"""
212. Word Search II
Given an m x n board of characters and a list of strings words, return all words on the board.

Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.

Input: board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]
"""


class TrieNode:
    def __init__(self):
        self.children = {}
        self.refs = 0
        self.isWord = False

    def addWord(self, word):
        cur = self
        self.refs += 1
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
            cur.refs += 1
        cur.isWord = True

    def removeWord(self, word):
        cur = self
        self.refs -= 1
        for c in word:
            if c in cur.children:
                cur = cur.children[c]
                cur.refs -= 1


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        root = TrieNode()
        for w in words:
            root.addWord(w)

        ROW, COL = len(board), len(board[0])
        res, visit = set(), set()

        def dfs(r, c, node, word):
            if (
                r not in range(ROW)
                or c not in range(COL)
                or board[r][c] not in node.children
                or node.children[board[r][c]].refs < 1
                or ((r, c)) in visit
            ):
                return
            visit.add((r, c))
            node = node.children[board[r][c]]
            word += board[r][c]
            if node.isWord:
                node.isWord = False
                res.add(word)
                root.removeWord(word)
            dfs(r + 1, c, node, word)
            dfs(r - 1, c, node, word)
            dfs(r, c + 1, node, word)
            dfs(r, c - 1, node, word)
            visit.remove((r, c))

        for r in range(ROW):
            for c in range(COL):
                dfs(r, c, root, "")
        return list(res)


# ----------------- Heap/Priority Queue -----------------
"""
295. Find Median from Data Stream
The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value, and the median is the mean of the two middle values.

For example, for arr = [2,3,4], the median is 3.
For example, for arr = [2,3], the median is (2 + 3) / 2 = 2.5.
Implement the MedianFinder class:

MedianFinder() initializes the MedianFinder object.
void addNum(int num) adds the integer num from the data stream to the data structure.
double findMedian() returns the median of all elements so far. Answers within 10-5 of the actual answer will be accepted.
 

Example 1:

Input
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
Output
[null, null, null, 1.5, null, 2.0]

Explanation
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0
"""


class MedianFinder:
    def __init__(self):
        # two heaps, large, small, minheap, maxheap
        # heaps should be equal size
        self.small, self.large = [], []  # maxHeap, minHeap (python default)

    def addNum(self, num: int) -> None:
        if self.large and num > self.large[0]:
            heapq.heappush(self.large, num)
        else:
            heapq.heappush(self.small, -1 * num)
        if len(self.small) > len(self.large) + 1:
            val = -1 * heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        if len(self.large) > len(self.small) + 1:
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -1 * val)

    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return -1 * self.small[0]
        elif len(self.large) > len(self.small):
            return self.large[0]
        return (-1 * self.small[0] + self.large[0]) / 2.0


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()


# ----------------- Backtracking -----------------
"""
Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.

The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the 
frequency
 of at least one of the chosen numbers is different.

The test cases are generated such that the number of unique combinations that sum up to target is less than 150 combinations for the given input.

Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.

"""


class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []

        def dfs(i, cur, total):
            if total == target:
                res.append(cur.copy())
            if i >= len(candidates) or total > target:
                return

            cur.append(candidates[i])
            dfs(
                i, cur, total + candidates[i]
            )  # call dfs with total + current candidate
            cur.pop()
            dfs(i + 1, cur, total)  # move to next index

        dfs(0, [], 0)
        return res


# 2d matrix + dfs + backtracking
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        ROWS, COLS = len(board), len(board[0])
        path = set()  # visited

        def dfs(r, c, i):
            if i == len(word):
                return True
            if (
                min(r, c) < 0
                or r >= ROWS
                or c >= COLS
                or word[i] != board[r][c]
                or (r, c) in path
            ):
                return False
            path.add((r, c))
            res = (
                dfs(r + 1, c, i + 1)
                or dfs(r - 1, c, i + 1)
                or dfs(r, c + 1, i + 1)
                or dfs(r, c - 1, i + 1)
            )
            path.remove((r, c))
            return res

        # to prevent TLE, reverse the word if frequency of the first letter is more than the last letter's
        count = defaultdict(int, sum(map(Counter, board), Counter()))
        if count[word[0]] > count[word[-1]]:
            word = word[::-1]
        for r in range(ROWS):
            for c in range(COLS):
                if dfs(r, c, 0):
                    return True
        return False


# ----------------- Graphs -----------------
"""
Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

Example 1:

Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
Example 2:

Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3

"""


class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        rows, cols = len(grid), len(grid[0])
        visit = set()

        def dfs(r, c, visited):
            if (
                r not in range(rows)
                or c not in range(cols)
                or (r, c) in visited
                or grid[r][c] == "0"
            ):
                return 0

            visited.add((r, c))
            # check 4 direction
            dfs(r + 1, c, visited)
            dfs(r - 1, c, visited)
            dfs(r, c + 1, visited)
            dfs(r, c - 1, visited)

            return 1

        islands_connected = 0
        for r in range(rows):
            for c in range(cols):
                islands_connected += dfs(r, c, visit)
        return islands_connected


# q2

"""
Given a reference of a node in a connected undirected graph.

Return a deep copy (clone) of the graph.
"""

"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

from typing import Optional


class Solution:
    def cloneGraph(self, node: Optional["Node"]) -> Optional["Node"]:
        oldToNew = {}

        def dfs(node):
            # check if node is in old dict
            if node in oldToNew:
                return oldToNew[node]
            copy = Node(node.val)  # make a copy
            oldToNew[node] = copy  # add copy to old dict

            for nei in node.neighbors:  # check neighbor
                copy.neighbors.append(
                    dfs(nei)
                )  # go through copy neighbors append dfs(nei)
            return copy

        return dfs(node) if node else None


"""
Return a 2D list of grid coordinates result where result[i] = [ri, ci] denotes that rain water can flow from cell (ri, ci) to both the Pacific and Atlantic oceans.


PacificAtlantic Approach:
- Do normal dfs: check currentHeight is greater than prevHeight
- Traverse the top and bottom rows to mamrk cells reachable from the Pacific and atlantic oceans
- Traverse the leftmost and rightmost columns to mark cells reachable from the Pacific and Atlantic oceans
- Find the intersection of cells reachable from both oceans.
"""


class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        ROWS, COLS = len(heights), len(heights[0])
        pac, atl = set(), set()

        # Normal graph DFS function
        def dfs(r, c, visit, prevHeight):
            # Base cases for stopping the DFS recursion
            if (
                (r, c) in visit
                or r < 0
                or c < 0
                or r == ROWS
                or c == COLS
                or heights[r][c] < prevHeight
            ):
                return
            # Mark the current cell as visited
            visit.add((r, c))
            # Recursively explore neighboring cells
            dfs(r + 1, c, visit, heights[r][c])
            dfs(r - 1, c, visit, heights[r][c])
            dfs(r, c + 1, visit, heights[r][c])
            dfs(r, c - 1, visit, heights[r][c])

        # Traverse the top and bottom rows to mark cells reachable from the Pacific and Atlantic oceans
        for c in range(COLS):
            dfs(0, c, pac, heights[0][c])  # Column in the Pacific
            dfs(ROWS - 1, c, atl, heights[ROWS - 1][c])  # Column in the Atlantic

        # Traverse the leftmost and rightmost columns to mark cells reachable from the Pacific and Atlantic oceans
        for r in range(ROWS):
            dfs(r, 0, pac, heights[r][0])  # Row in the Pacific
            dfs(r, COLS - 1, atl, heights[r][COLS - 1])  # Row in the Atlantic

        # Find the intersection of cells reachable from both oceans
        res = []
        for r in range(ROWS):
            for c in range(COLS):
                if (r, c) in pac and (r, c) in atl:
                    res.append([r, c])

        return res

"""
207. Course Schedule
Approach:
- convert prerequisites to adjacency list
- DFS(crs) -> check if crs in visiting -> if preMap crs is done -> add visiting crs -> check neighbors of cours -> remove visited crs -> return True
- iterate through number of courses if not dfs return False
- finally return True
"""
"""
{
    0: 1
    1: 0
}
"""
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # convert prerequisites to adjacency list
        preMap = {i: [] for i in range(numCourses)} # {0: [], 1:[] }
        visiting = set()

        for crs, pre in prerequisites: # {0: [], 1: [0]}
            preMap[crs].append(pre)
       
        # DFS(crs) -> check if crs in visiting -> if preMap crs is done -> add visiting crs -> check neighbors of course -> remove visited crs -> return True
        def dfs(crs):
            if crs in visiting:
                return False 
            if preMap[crs] == []:
                return True
            visiting.add(crs)

            for pre in preMap[crs]:
                if not dfs(pre):
                    return False

            visiting.remove(crs)
            preMap[crs] = []
            return True

        # Iterate through number of courses if not dfs return False
        for crs in range(numCourses):
            if not dfs(crs):
                return False
        # finally return True
        return True
# ----------------- Advanced Graphs -----------------
    # check 150 or neetcode all

# ----------------- 1-D Dynamic Programming -----------------
"""
Climbing Stairs
You are climbing a staircase. It takes n steps to reach the top.
Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

There are two ways to climb to the top
1. 1 step + 1 step
2. 2 steps
Approach:
- Base cases: If if the number of steps is 1,2, or 3 return respective value
- initialize variables for the number of distinct ways to climb stairs with 2 and 3 steps
- Iterate from 4 to n (inclusive) to calculate distinct ways for each step count
    - caclculate the current distinct ways by adding the values for n1 and n2
    - update n1 and n2 for the next iteration
- Return the total number of distinct ways to climb the staircase with n steps
"""
class Solution:
    def climbStairs(self, n: int) -> int:
        # - Base cases: If the number of steps is 1,2, or 3 return respective value
        if n <= 3:
            return n
        
        # - initialize variables for the number of distinct ways to climb stairs with 2 and 3 steps
        n1, n2 = 2, 3
        # - Iterate from 4 to n (inclusive) to calculate distinct ways for each step count
        for i in range(4, n + 1):
            # caclculate the current distinct ways by adding the values for n1 and n2
            temp = n1 + n2
            # update n1 and n2 for the next iteration
            n1 = n2
            n2 = temp
        # - Return the total number of distinct ways to climb the staircase with n steps
        return n2

# ----------------- 2-D Dynamic Programming -----------------
# ----------------- Greedy -----------------
# ----------------- Intervals -----------------
# ----------------- Math & Geometry -----------------
# ----------------- Bit Manipulation -----------------


"""
ChatGPT code explanation prompt:
- add inline code explaining this code
"""
