class Solution(object):
    res = []

    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        self.allRange(0, nums)
        return self.res

    def allRange(self, begin, nums):
        if begin == len(nums) - 1:
            import copy
            temp = copy.copy(nums)
            self.res.append(temp)
        else:
            for i in range(begin, len(nums)):
                nums[i], nums[begin] = nums[begin], nums[i]
                self.allRange(begin + 1, nums)
                nums[i], nums[begin] = nums[begin], nums[i]


if __name__ == '__main__':
    s = Solution()
    arr = [1, 2]
    print s.permute(arr)
