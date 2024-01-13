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


# ----------------- Sliding Window -----------------
# ----------------- Stack -----------------
# ----------------- Binary Search -----------------
# ----------------- Linked List -----------------
# ----------------- Trees -----------------
# ----------------- Tries -----------------
# ----------------- Heap/Priority Queue -----------------
# ----------------- Backtracking -----------------
# ----------------- Graphs -----------------
# ----------------- Advanced Graphs -----------------
# ----------------- 1-D Dynamic Programming -----------------
# ----------------- 2-D Dynamic Programming -----------------
# ----------------- Greedy -----------------
# ----------------- Intervals -----------------
# ----------------- Math & Geometry -----------------
# ----------------- Bit Manipulation -----------------
