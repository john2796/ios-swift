class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = {} #{1:3, 2:2, 3:1}
        freq = [[] for i in range(len(nums) + 1)] # [[], [], [], []]
        for n in nums:
            count[n] = 1 + count.get(n, 0)
        for key, value in count.items():
            freq[value].append(key)

        # print(freq) #  [[], [3], [2], [1], [], [], []]
        res = []
        for i in range(len(freq) - 1, 0, -1): # # loop through freq in reverse, then store values in res, once k equals res return result. 
            for n in freq[i]:
                res.append(n)
                if k == len(res):
                    return res




