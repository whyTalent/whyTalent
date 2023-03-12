# N sum问题

**题目**：有一n个数值的数组 nums，返回所有满足N个数组元素和为 target的元素列表，且每个列表是唯一的，不重复

​      

1）[2 sum](https://leetcode.com/problems/two-sum/)

```java
输入: nums = [2,7,11,15], target = 9
输出: [0,1]
说明: nums[0] + nums[1] == 9, 返回 [0, 1].
```

​    

2）[3 sum](https://leetcode.com/problems/3sum/)

```java
输入: nums = [-1,0,1,2,-1,-4]
输出: [[-1,-1,2],[-1,0,1]]
说明: 
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
但结果列表不一致的是 [-1,0,1] and [-1,-1,2]
```

​      

3）[4 sum](https://leetcode.com/problems/4sum/)

 `[nums[a], nums[b], nums[c], nums[d]]` :

- `0 <= a, b, c, d < n`
- `a`, `b`, `c`, 和 `d` 是不同的.
- `nums[a] + nums[b] + nums[c] + nums[d] == target`

​    

## 核心算法

思路：求解方式就是在 “**求解 N−1个数之和等于 target值**” 基础上，再加一层循环。最底层的 `2Sum` 的复杂度可以降低到 `O(L)`，所以 `NSum` 的复杂度是 `O(L^(N−1))`，L 是数组长度。

​    

### 1）回溯

定义回溯方法 `backtrack(res,path,idx,target)` ，即从数组 `nums` 的下标 `idx` 开始，寻找和值等于 `target`  的组合，把组合添加到结果集 `res` 中，`path` 是回溯过程中的选择路径。

回溯过程：

> 结束条件：路径长度等于 N 时：
>
> - 如果满足 target==0，表示已经找到一个组合，就把选择路径添加进结果集；
> - 返回上一个回溯点；
>
> 选择路径：记录已经选中的数字，作为组合中的元素；
>
> 选择：当前数字不是该层递归中第一个元素，且与数组中前一个元素值不相等，就添加进选择列表中；
>
> 空间状态树：略

```go
func nSum(nums []int, target int, N int) [][]int {
  // sort
	sort.Ints(nums)

	results := [][]int{}
	result := []int{}
	backtrack(nums, target, N, 0, &result, &results)

	return results
}

func backtrack(nums []int, target int, N int, idx int, result *[]int, results *[][]int) {
	if len(*result) == N {
		if target == 0 {
      // tips: 由于函数传递到话slice地址, 因此不能将result赋值给其它变量, 比如 tmp := *result
			tmp := make([]int, N)
			copy(tmp, *result)
			
			*results = append(*results, tmp)
		}

		return
	}
	
  // 回溯法
	for i := idx; i < len(nums); i++ {
		if i > idx && nums[i] == nums[i-1] {
			continue
		}

		*result = append(*result, nums[i])
		backtrack(nums, target-nums[i], N, i+1, result, results)
		*result = (*result)[:len(*result)-1]
	}
}
```



### 2）递归

递归方法：`recursion(n,target,start)`，即在数组 `nums` 的下标 `start` 开始，寻找 `n` 个数，要求它们的和值等于 `target`。递归的结束条件是：当传入的  `n==2` 时，这时使用 `2Sum` 的求解方法

```go
import (
	"sort"
)

func nSum(nums []int, target int, N int) [][]int {
  // sort
	sort.Ints(nums)

	results := [][]int{}
  // tips: go内函数参数为值传递, 因此函数参数应该传入slice的地址, 否则函数内部对slice操作不会传递给调用方
	recursion(nums, target, N, []int{}, &results)

	return results
}

func recursion(nums []int, target int, N int, result []int, results *[][]int) {
  n := len(nums)
	if n < N || N < 2 {
		return
	}

	if N == 2 {  // 2-sum
		l, r := 0, n-1
		for l < r {
			if nums[l] + nums[r] == target {
        // result = append(result, nums[l], nums[r]); *results = append(*results, result)
        // tips: 上述赋值方式当前缀一致, 但后续列表中存在多个满足条件的子列表时，结果异常, 比如 [-3, 0, 0, 1, 2, 3]
				*results = append(*results, append(result, nums[l], nums[r]))
				
				l++
				r--
				for l < r && nums[l] == nums[l-1] {
					l++
				}
				for r > l && nums[r] == nums[r+1] {
					r--
				}
			} else if nums[l] + nums[r] < target {
				l++
			} else {
				r--
			}
		}
	} else {  // N-sum 问题逐步递归到 2-sum 问题
		for i := 0; i < n-N+1; i++ {
      // tips: 根据有序数组特性, 排除最小值&最大值
			if target < N*nums[i] || target > nums[n-1]*N {
				break
			}

			if i == 0 || (i > 0 && nums[i-1] != nums[i]) {
				recursion(nums[i+1:], target-nums[i], N-1, append(result, nums[i]), results)
			}
		}
	}
}
```















