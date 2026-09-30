# Failure analysis: worked examples

From the official run: 77 of 3765 test queries (2.0%) do not place the relevant document in the top 10.

| Group | Queries | Share of failures |
|---|---|---|
| other | 56 | 72.7% |
| near_duplicate_corpus | 14 | 18.2% |
| generic_wording | 7 | 9.1% |

## Example 1: other (relevant document at rank not retrieved)

**Why it is grouped here:** 100 query terms, best overlap with a retrieved document 33% -- no obvious cause

### Query

```
Recall that a binary search tree is a rooted binary tree, whose nodes each store a key and each have at most two distinguished subtrees, left and right. The key in each node must be greater than any key stored in the left subtree, and less than any key stored in the right subtree.

The depth of a vertex is the number of edges on the simple path from the vertex to the root. In particular, the depth of the root is $0$.

Let's call a binary search tree perfectly balanced if there doesn't exist a binary search tree with the same number of vertices that has a strictly smaller sum of depths of its vertices.

Let's call a binary search tree with integer keys striped if both of the following conditi
```

### Expected document (`d5155`)

```python
N = int(input())
if N in [1, 2, 4, 5, 9, 10, 20, 21, 41, 42, 84, 85, 169, 170, 340, 341, 681, 682, 1364, 1365, 2729, 2730, 5460, 5461, 10921, 10922, 21844, 21845, 43689, 43690, 87380, 87381, 174761, 174762, 349524, 349525, 699049, 699050]:
    print(1)
else:
    print(0)
```

### Retrieved #1 (`d5881`)

```python
MOD = 998244353

list_size = 1000001

f_list = [1] * list_size
f_r_list = [1] * list_size

for i in range(list_size - 1):
	f_list[i + 1] = int((f_list[i] * (i + 2)) % MOD)

def power(n, x):
	if x == 1:
		return n
	elif x % 2 == 0:
		return power(int((n * n) % MOD), int(x / 2))
	else:
		return int((n * power(n, x - 1)) % MOD)

f_r_list[-1] = power(f_list[-1], MOD - 2)

for i in range(2, list_size + 1):
	f_r_list[-i] = int((f_r_list[-i + 1] * (list_size + 2 - i)) % MOD)

def comb(n, r):
	if n < r:
		return 0
	elif n == 0 or r == 0 or n == r:
		return 1
	else:
		return (((f_list[n - 1] * f_r_list[n - r - 1]) % MOD) * f_r_list[r - 1]) % MOD 

n = int(input())
ans = f_list[n-1]
for i in range(2,
```

### Retrieved #2 (`d7902`)

```python
N=int(input())
mod=998244353
inv4=249561088
A=[inv4,0,3]
for i in range(N):
    A.append(9*A[-1]-24*A[-2]+16*A[-3])
    A[-1]%=mod
A[0]-=inv4
B=[0 for i in range(N)]
for i in range(N):
    x=i
    y=N-i-1
    if x<=y:
        B[x]=A[i]
        B[y]=A[i]

P=N*pow(2,N-2+mod-1,mod)
for i in range(N):
    B[i]+=P
    B[i]%=mod
Q=pow(2,N-1,mod)
Qinv=pow(Q,mod-2,mod)
for i in range(N):
    B[i]*=Qinv
    B[i]%=mod
for i in range(N):
    print((B[i]))
```

### Retrieved #3 (`d2357`)

```python
n,k=map(int,input().split())
mod=998244353
dp=[0]*(n+1)
dp[0]=1
for i in range(1,n+1):
  for j in range(i,-1,-1):
    dp[j]=dp[j-1]
    if 2*j<=i:
      dp[j]+=dp[2*j]
    dp[j]%=mod
print(dp[k])
```

## Example 2: near_duplicate_corpus (relevant document at rank 15)

**Why it is grouped here:** a retrieved document shares 71% of its identifiers with the expected one: the corpus holds near-identical solutions

### Query

```
You are given an integer N. Consider all possible segments on the coordinate axis with endpoints at integer points with coordinates between 0 and N, inclusive; there will be $\frac{n(n + 1)}{2}$ of them.

You want to draw these segments in several layers so that in each layer the segments don't overlap (they might touch at the endpoints though). You can not move the segments to a different location on the coordinate axis. 

Find the minimal number of layers you have to use for the given N.


-----Input-----

The only input line contains a single integer N (1 ≤ N ≤ 100).


-----Output-----

Output a single integer - the minimal number of layers required to draw the segments for the given N.
```

### Expected document (`d5090`)

```python
n=int(input())
print(max((i+1)*(n-i)for i in range(n)))
```

### Retrieved #1 (`d8636`)

```python
N = int(input())
 
ans = int(N*(N+1)/2)
print(ans)
```

### Retrieved #2 (`d6046`)

```python
n = int(input())
ans = 0
s = 0

while (n > 0):
    ans += 1
    s += ans
    n -= s

    if (n < 0):
        ans -= 1
        break

print(ans)
```

### Retrieved #3 (`d8097`)

```python
N = int(input())
ANS = 1
for i in range(1,N-1):
  ANS += (N-1)//i
  
print(ANS)
```

## Example 3: generic_wording (relevant document at rank not retrieved)

**Why it is grouped here:** only 14 distinct terms in the query; many documents are plausible answers

### Query

```
[Image] 


-----Input-----

The input contains two integers a_1, a_2 (0 ≤ a_{i} ≤ 32), separated by a single space.


-----Output-----

Output a single integer.


-----Examples-----
Input
1 1

Output
0

Input
3 7

Output
0

Input
13 10

Output
1
```

### Expected document (`d5696`)

```python
import math
import re
from fractions import Fraction

class Task:
    table = ['111111101010101111100101001111111\n', '100000100000000001010110001000001\n', '101110100110110000011010001011101\n', '101110101011001001111101001011101\n', '101110101100011000111100101011101\n', '100000101010101011010000101000001\n', '111111101010101010101010101111111\n', '000000001111101111100111100000000\n', '100010111100100001011110111111001\n', '110111001111111100100001000101100\n', '011100111010000101000111010001010\n', '011110000110001111110101100000011\n', '111111111111111000111001001011000\n', '111000010111010011010011010100100\n', '101010100010110010110101010000010\n', '101100000101010001111101000000000\n
```

### Retrieved #1 (`d8253`)

```python
a,b=map(int,input().split())
if a > 2*b:
  print(a-2*b)
else:
  print("0")
```

### Retrieved #2 (`d3921`)

```python
def hamming_distance(a, b):
    return bin(a ^ b).count('1')
```

### Retrieved #3 (`d2659`)

```python
def convert_bits(a,b):
    return bin(a^b).count("1")
```

## Example 4: other (relevant document at rank not retrieved)

**Why it is grouped here:** 70 query terms, best overlap with a retrieved document 0% -- no obvious cause

### Query

```
You are at a water bowling training. There are l people who play with their left hand, r people, who play with their right hand, and a ambidexters, who can play with left or right hand.

The coach decided to form a team of even number of players, exactly half of the players should play with their right hand, and exactly half of the players should play with their left hand. One player should use only on of his hands.

Ambidexters play as well with their right hand as with their left hand. In the team, an ambidexter can play with their left hand, or with their right hand.

Please find the maximum possible size of the team, where equal number of players use their left and right hands, respectiv
```

### Expected document (`d5185`)

```python
import base64
import zlib
pro = base64.decodebytes("""eJxtUUFuwyAQvPOKVarKkDhOm2MlX/uC3qqqAhs7KBgswGr6+y4QrLqqL7DD7OzMWk2zdQFGGWbu
PVG59N/rdeLhUu6Om95OpVJBumCtXqlCedkFQgalpYcW3twiSS/FMmLxyrWXhKihzGrwXLx0lEHb
QjU4e5HmWgHOgKTwQgC/0p/EIoDeGh96ZRC0szR0F6QPjTI7lt4fCsMuoVCqREGgqqH6qjIxBSZo
cADdTZTXIFie6dCZM8BhDwJOp7SDZuz6zLn3OMXplv+uTKCKwWAdKECDysxLoKzxs1Z4fpRObkb5
6ZfNTDSDbimlAo44+QDPLI4+MzRBYy1Yto0bxPqINTzCOe7uKSsUlQPKFJFzFtmkWlN3dhKcmhpu
2xw05R14FyyG1NSwdQm/QJxwY/+93OKGdA2uRgtt3hPp1RALLjzV2OkYmZSJCB40ku/AISORju2M
XOEPkISOLVzJ/ShtPCedXfwLCdxjfPIDQSHUSQ==
""".encode())
pro = zlib.decompress(pro)
pro = pro.decode()
exec(pro)
```

### Retrieved #1 (`d8340`)

```python
a, b = map(int, input().split())
print(max(a+a-1, a+b, b+b-1))
```

### Retrieved #2 (`d5442`)

```python
n, a, b = list(map(int, input().split()))
s = input()
s += '*'
n += 1
m = []
i = 0
i1 = -1
while i < len(s):
    if s[i] == '*':
        if i - i1 > 1:
            m.append(i - i1 - 1)
        i1 = i
    i += 1
sm = a + b
for c in m:
    if c % 2 == 0:
        a = max(0, a - c // 2)
        b = max(0, b - c // 2)
    else:
        if a > b:
            a = max(0, a - (c + 1) // 2)
            b = max(0, b - c // 2)
        else:
            b = max(0, b - (c + 1) // 2)
            a = max(0, a - c // 2)
print(sm - a - b)
```

### Retrieved #3 (`d8650`)

```python
a, b = map(int, input().split())
print(((a + b) + (2 - 1)) // 2)
```

## Example 5: near_duplicate_corpus (relevant document at rank 19)

**Why it is grouped here:** a retrieved document shares 60% of its identifiers with the expected one: the corpus holds near-identical solutions

### Query

```
Pasha has a wooden stick of some positive integer length n. He wants to perform exactly three cuts to get four parts of the stick. Each part must have some positive integer length and the sum of these lengths will obviously be n. 

Pasha likes rectangles but hates squares, so he wonders, how many ways are there to split a stick into four parts so that it's possible to form a rectangle using these parts, but is impossible to form a square.

Your task is to help Pasha and count the number of such ways. Two ways to cut the stick are considered distinct if there exists some integer x, such that the number of parts of length x in the first way differ from the number of parts of length x in the se
```

### Expected document (`d5199`)

```python
x = int(input())
if x%2==1:
    print(0)
    quit()
if x%2 ==0:
    x//=2
    if x%2==0:
        print(x//2-1)
    else:
        print(x//2)
```

### Retrieved #1 (`d8086`)

```python
import math
n=int(input())
ans=0
num=int(math.sqrt(n)+1)
for i in range(1,num)[::-1]:
    if n%i==0:
        ans=i+(n//i)-2
        break
print(ans)
```

### Retrieved #2 (`d8385`)

```python
from math import *

n = int(input())
print(factorial(n) // (factorial(n // 2) ** 2) * factorial(n//2-1) ** 2 // 2)
```

### Retrieved #3 (`d6101`)

```python
n = int(input())
print((n-2)**2)
```
