# Day 01 - Python Core Concepts: Comprehensive Guide

**Topics Covered:** Python Fundamentals, Memory Management, Data Structures, Performance Optimization

---

## Question 1: Mutable vs Immutable Objects

**Topic:** Python  
**Difficulty:** Intermediate

### Question
What is the difference between mutable and immutable objects in Python? Give examples and explain the implications.

### Answer

Immutable objects cannot be changed after creation. Any modification creates a new object. Mutable objects can be modified in place without creating a new object.

```python
# Immutable: int, float, string, tuple, frozenset
x = 10
print(id(x))  # Memory address: 140712345678912
x = x + 5
print(id(x))  # Different address: 140712345678992 (new object created)

s = "hello"
print(id(s))  # Address: 2234567891234
s = s + " world"
print(id(s))  # Different address: 2234567891456 (new string object)

# Mutable: list, dict, set
my_list = [1, 2, 3]
print(id(my_list))  # Address: 2234567892345
my_list.append(4)
print(id(my_list))  # Same address: 2234567892345 (modified in place)

my_dict = {'a': 1}
print(id(my_dict))  # Address: 2234567893456
my_dict['b'] = 2
print(id(my_dict))  # Same address: 2234567893456 (modified in place)
```

**Key Implications:**
- Immutable objects are hashable and can be dictionary keys
- Mutable objects are more memory-efficient for frequent modifications
- Immutable objects are thread-safe

---

## Question 2: Memory Management in Python

**Topic:** Python  
**Difficulty:** Intermediate

### Question
How is memory management handled in Python?

### Answer

Python uses automatic memory management with several key mechanisms:

**Reference Counting:**
```python
import sys

a = [1, 2, 3]
print(sys.getrefcount(a))  # 2 (one from 'a', one from getrefcount argument)

b = a  # Creates another reference
print(sys.getrefcount(a))  # 3

del b  # Removes one reference
print(sys.getrefcount(a))  # 2
```

**Garbage Collection:**
Handles circular references that reference counting can't detect.

```python
import gc

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

# Create circular reference
node1 = Node(1)
node2 = Node(2)
node1.next = node2
node2.next = node1  # Circular!

# Delete references
del node1, node2  # Reference counting can't clean this up

# Garbage collector detects and cleans circular references
gc.collect()  # Returns number of objects collected
```

**Memory Pools:**
Python uses private heaps and memory pools for efficient allocation of small objects.

```python
# Small integers (-5 to 256) are cached
a = 10
b = 10
print(a is b)  # True (same object)

# Larger integers are not
x = 1000
y = 1000
print(x is y)  # False (different objects)
```

---

## Question 3: `is` vs `==` Operators

**Topic:** Python  
**Difficulty:** Intermediate

### Question
Explain the difference between `is` and `==` operators with examples.

### Answer

`==` compares values (calls `__eq__` method), `is` compares identity (memory addresses).

```python
# Example 1: Lists with same values
list1 = [1, 2, 3]
list2 = [1, 2, 3]

print(list1 == list2)  # True (same values)
print(list1 is list2)  # False (different objects in memory)
print(id(list1), id(list2))  # Different addresses

# Example 2: Same object reference
list3 = list1
print(list1 is list3)  # True (same object)
print(list1 == list3)  # True (same values)

# Example 3: Singleton None
a = None
b = None
print(a is b)  # True (None is a singleton)
print(a == b)  # True (equal values)

# Example 4: String interning (optimization)
s1 = "hello"
s2 = "hello"
print(s1 is s2)  # True (strings are interned)

s3 = "hello world"
s4 = "hello world"
print(s3 is s4)  # May be True or False (implementation dependent)

# Example 5: Custom comparison
class Person:
    def __init__(self, name):
        self.name = name
    
    def __eq__(self, other):
        return self.name == other.name

p1 = Person("Alice")
p2 = Person("Alice")
print(p1 == p2)  # True (same name)
print(p1 is p2)  # False (different objects)
```

---

## Question 4: Shallow Copy vs Deep Copy

**Topic:** Python  
**Difficulty:** Intermediate

### Question
What is the difference between a shallow copy and a deep copy in Python?

### Answer

Shallow copy creates a new object but references the same nested objects. Deep copy creates a new object and recursively copies all nested objects.

```python
import copy

# Example 1: Simple list (no difference)
original = [1, 2, 3]
shallow = original.copy()
deep = copy.deepcopy(original)

shallow[0] = 99
print(original)  # [1, 2, 3] (unchanged)
print(shallow)   # [99, 2, 3]

# Example 2: Nested list (shows difference)
original = [[1, 2, 3], [4, 5, 6]]

shallow = original.copy()
deep = copy.deepcopy(original)

# Modify nested list in shallow copy
shallow[0][0] = 999
print(original)  # [[999, 2, 3], [4, 5, 6]] (CHANGED! References same nested list)
print(shallow)   # [[999, 2, 3], [4, 5, 6]]
print(deep)      # [[1, 2, 3], [4, 5, 6]] (unchanged)

# Example 3: Dictionary with nested objects
original_dict = {
    'name': 'Alice',
    'scores': [85, 90, 95],
    'metadata': {'age': 25}
}

shallow_dict = original_dict.copy()
deep_dict = copy.deepcopy(original_dict)

# Modify nested structures
shallow_dict['scores'].append(100)
shallow_dict['metadata']['age'] = 30

print(original_dict['scores'])  # [85, 90, 95, 100] (modified!)
print(original_dict['metadata'])  # {'age': 30} (modified!)
print(deep_dict['scores'])  # [85, 90, 95] (unchanged)
print(deep_dict['metadata'])  # {'age': 25} (unchanged)
```

---

## Question 5: Data Structure Performance

**Topic:** Python  
**Difficulty:** Intermediate

### Question
How do Python's list, tuple, set, and dictionary differ in terms of performance and usage?

### Answer

| Data Structure | Mutable | Ordered | Duplicates | Access | Search | Usage |
|---------------|---------|---------|------------|--------|--------|-------|
| List | Yes | Yes | Yes | O(1) | O(n) | Sequential data, frequent modifications |
| Tuple | No | Yes | Yes | O(1) | O(n) | Fixed data, dictionary keys, unpacking |
| Set | Yes | No | No | N/A | O(1) | Membership testing, unique items |
| Dictionary | Yes | Yes* | No (keys) | O(1) | O(1) | Key-value pairs, fast lookups |

*Ordered since Python 3.7

```python
import time

# List: Best for sequential access and modifications
my_list = [1, 2, 3, 4, 5]
my_list.append(6)  # O(1)
my_list[2]  # O(1) access by index
my_list.insert(0, 0)  # O(n) - shifts elements

# Tuple: Immutable, faster iteration, less memory
my_tuple = (1, 2, 3, 4, 5)
x, y, z = (10, 20, 30)  # Unpacking
coordinates = {(0, 0): 'origin', (1, 1): 'point'}  # As dict keys

# Set: Fast membership testing
data = [1, 2, 2, 3, 3, 3, 4, 5]
unique = set(data)  # {1, 2, 3, 4, 5}

# Performance comparison: membership testing
test_list = list(range(10000))
test_set = set(range(10000))

start = time.time()
9999 in test_list  # O(n) - checks each element
list_time = time.time() - start

start = time.time()
9999 in test_set  # O(1) - hash lookup
set_time = time.time() - start

print(f"List: {list_time:.6f}s, Set: {set_time:.6f}s")
# Set is ~1000x faster for membership testing

# Dictionary: Fast key-based access
user_data = {
    'user123': {'name': 'Alice', 'age': 30},
    'user456': {'name': 'Bob', 'age': 25}
}
print(user_data['user123'])  # O(1) lookup

# Set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}
print(set1 & set2)  # Intersection: {3, 4}
print(set1 | set2)  # Union: {1, 2, 3, 4, 5, 6}
print(set1 - set2)  # Difference: {1, 2}
```

---

## Question 6: Built-in Data Structures

**Topic:** Python  
**Difficulty:** Intermediate

### Question
What are Python's built-in data structures, and when would you prefer one over another?

### Answer

```python
# List: Use when order matters and you need modifications
tasks = []
tasks.append("Write code")
tasks.append("Test code")
tasks.insert(1, "Review code")
tasks.remove("Test code")

# Tuple: Use for fixed collections, function returns, dictionary keys
def get_coordinates():
    return (10, 20)  # Return multiple values

x, y = get_coordinates()

point_values = {
    (0, 0): 100,
    (1, 1): 200
}  # Tuples as immutable keys

# Set: Use for uniqueness, membership testing, set operations
visited_urls = set()
visited_urls.add("https://example.com")
if "https://example.com" in visited_urls:  # O(1) check
    print("Already visited")

# Remove duplicates from list
numbers = [1, 2, 2, 3, 3, 3, 4, 5, 5]
unique_numbers = list(set(numbers))  # [1, 2, 3, 4, 5]

# Dictionary: Use for key-value associations
user_scores = {
    'Alice': 95,
    'Bob': 87,
    'Charlie': 92
}

# Cache/memoization
cache = {}
def fibonacci(n):
    if n in cache:
        return cache[n]
    if n <= 1:
        return n
    cache[n] = fibonacci(n-1) + fibonacci(n-2)
    return cache[n]

# Counter (from collections)
from collections import Counter, defaultdict, deque

# Counter: Frequency counting
words = ['apple', 'banana', 'apple', 'orange', 'banana', 'apple']
word_count = Counter(words)
print(word_count)  # Counter({'apple': 3, 'banana': 2, 'orange': 1})

# defaultdict: Auto-initializing values
groups = defaultdict(list)
groups['fruits'].append('apple')  # No KeyError if 'fruits' doesn't exist

# deque: Efficient queue operations (O(1) on both ends)
queue = deque([1, 2, 3])
queue.append(4)  # Add to right: O(1)
queue.appendleft(0)  # Add to left: O(1)
queue.pop()  # Remove from right: O(1)
queue.popleft()  # Remove from left: O(1)
```

---

## Question 7: String Storage and Immutability

**Topic:** Python  
**Difficulty:** Intermediate

### Question
How are strings stored internally in Python, and why are they immutable?

### Answer

Strings in Python are stored as arrays of Unicode code points and are immutable for several reasons:

**Internal Storage:**
```python
import sys

# Strings are stored as contiguous arrays
s = "Hello"
print(sys.getsizeof(s))  # Memory size in bytes

# Each character is a Unicode code point
print(ord('A'))  # 65 (ASCII/Unicode value)
print(chr(65))   # 'A'

# String interning (optimization)
a = "hello"
b = "hello"
print(a is b)  # True (same object in memory)

# Python interns strings that look like identifiers
s1 = "python_variable"
s2 = "python_variable"
print(s1 is s2)  # True

s3 = "hello world!"
s4 = "hello world!"
print(s3 is s4)  # May be False (not interned due to space and !)
```

**Why Immutability?**
```python
# 1. Hashability - strings can be dictionary keys
word_count = {"hello": 1, "world": 2}

# 2. Security - prevents unintended modifications
password = "secret123"
# If strings were mutable, functions could change passwords unexpectedly

# 3. Thread safety - multiple threads can read without locks
shared_config = "DATABASE_URL=localhost"

# 4. Optimization - string interning saves memory
names = ["Alice"] * 1000  # All refer to same string object

# String operations create new objects
s = "hello"
print(id(s))  # Address: 140234567890123

s = s + " world"  # Creates new string
print(id(s))  # Different address: 140234567890456

# Efficient string building with join (avoids creating many temp strings)
# Bad: O(n²) time complexity
result = ""
for i in range(1000):
    result += str(i)  # Creates new string each iteration

# Good: O(n) time complexity
result = "".join(str(i) for i in range(1000))

# Or use list accumulation
parts = []
for i in range(1000):
    parts.append(str(i))
result = "".join(parts)
```

---

## Question 8: Hashing in Sets and Dictionaries

**Topic:** Python  
**Difficulty:** Intermediate

### Question
Explain how hashing works in sets and dictionaries.

### Answer

Both sets and dictionaries use hash tables for O(1) average-case operations.

```python
# Hash function converts object to integer
print(hash("hello"))  # 5909654896763645359
print(hash(42))       # 42
print(hash((1, 2)))   # 3713081631934410656

# Hash must be consistent
s = "python"
print(hash(s) == hash(s))  # True (always same hash)

# Mutable objects cannot be hashed
try:
    hash([1, 2, 3])  # TypeError: unhashable type: 'list'
except TypeError as e:
    print(e)

# Dictionary internals (simplified)
class SimpleDict:
    def __init__(self):
        self.size = 8
        self.buckets = [[] for _ in range(self.size)]
    
    def _get_bucket(self, key):
        hash_value = hash(key)
        index = hash_value % self.size
        return self.buckets[index]
    
    def set(self, key, value):
        bucket = self._get_bucket(key)
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)  # Update existing
                return
        bucket.append((key, value))  # Add new
    
    def get(self, key):
        bucket = self._get_bucket(key)
        for k, v in bucket:
            if k == key:
                return v
        raise KeyError(key)

# Usage
d = SimpleDict()
d.set("name", "Alice")
d.set("age", 30)
print(d.get("name"))  # Alice

# Custom hash function
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __hash__(self):
        return hash((self.x, self.y))  # Combine hashable components
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

# Now Point objects can be dictionary keys/set members
point_set = {Point(1, 2), Point(3, 4), Point(1, 2)}
print(len(point_set))  # 2 (duplicate removed)

locations = {
    Point(0, 0): "origin",
    Point(10, 20): "destination"
}
print(locations[Point(0, 0)])  # "origin"
```

---

## Question 9: List Comprehension vs For Loop

**Topic:** Python  
**Difficulty:** Intermediate

### Question
What's the difference between a list comprehension and using a for loop for creating lists? Which one is faster and why?

### Answer

List comprehensions are faster and more readable for creating lists.

```python
import time
import dis

# Method 1: For loop
def for_loop_method():
    result = []
    for i in range(1000):
        result.append(i * 2)
    return result

# Method 2: List comprehension
def list_comp_method():
    return [i * 2 for i in range(1000)]

# Performance comparison
start = time.time()
for _ in range(10000):
    for_loop_method()
for_loop_time = time.time() - start

start = time.time()
for _ in range(10000):
    list_comp_method()
list_comp_time = time.time() - start

print(f"For loop: {for_loop_time:.4f}s")
print(f"List comp: {list_comp_time:.4f}s")
print(f"Speedup: {for_loop_time/list_comp_time:.2f}x")
# List comprehension is typically 20-30% faster

# Complex examples
# 1. With condition
evens = [x for x in range(20) if x % 2 == 0]

# 2. Nested loops
matrix = [[i * j for j in range(3)] for i in range(3)]
# [[0, 0, 0], [0, 1, 2], [0, 2, 4]]

# 3. Dictionary comprehension
squares = {x: x**2 for x in range(10)}

# 4. Set comprehension
unique_lengths = {len(word) for word in ['hello', 'world', 'hi', 'hey']}

# 5. Generator expression (lazy evaluation)
gen = (x**2 for x in range(1000000))  # No memory allocated yet
print(next(gen))  # 0 (computes on demand)
```

---

## Question 10: Object References and Garbage Collection

**Topic:** Python  
**Difficulty:** Intermediate

### Question
How does Python manage object references and garbage collection?

### Answer

Python uses reference counting with cyclic garbage collection to manage memory.

```python
import sys
import gc
import weakref

# Reference counting basics
a = [1, 2, 3]
print(sys.getrefcount(a))  # 2 (one from 'a', one from function call)

b = a  # Increment reference count
print(sys.getrefcount(a))  # 3

c = [a, a]  # Two more references
print(sys.getrefcount(a))  # 5

del b  # Decrement
print(sys.getrefcount(a))  # 4

# Circular references (reference counting can't handle)
class Node:
    def __init__(self, value):
        self.value = value
        self.ref = None
    
    def __del__(self):
        print(f"Node {self.value} deleted")

# Create circular reference
node1 = Node(1)
node2 = Node(2)
node1.ref = node2
node2.ref = node1  # Cycle!

# Delete our references
del node1
del node2
# Nodes not deleted yet - circular reference keeps count > 0

# Garbage collector detects and breaks cycles
print("Running garbage collector...")
collected = gc.collect()
print(f"Collected {collected} objects")
# Output: Node 1 deleted, Node 2 deleted

# Weak references (don't increase ref count)
class Cache:
    def __init__(self):
        self._cache = {}
    
    def add(self, key, obj):
        self._cache[key] = weakref.ref(obj)  # Weak reference
    
    def get(self, key):
        weak_ref = self._cache.get(key)
        if weak_ref:
            obj = weak_ref()  # Dereference
            if obj is not None:
                return obj
        return None

cache = Cache()
data = [1, 2, 3]
cache.add('data', data)

print(cache.get('data'))  # [1, 2, 3]

del data  # Object can be garbage collected
gc.collect()

print(cache.get('data'))  # None (object was collected)

# Memory management tips
# 1. Use __slots__ to reduce memory per instance
class SlottedClass:
    __slots__ = ['x', 'y']  # No __dict__, less memory
    def __init__(self, x, y):
        self.x = x
        self.y = y
```

---

## 📚 Additional Resources

- [Python Documentation - Data Structures](https://docs.python.org/3/tutorial/datastructures.html)
- [Python Memory Management](https://realpython.com/python-memory-management/)
- [Understanding Python's Execution Model](https://docs.python.org/3/reference/executionmodel.html)
- [Python Performance Tips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)

---

## 🎯 Key Takeaways

- Understand the difference between mutable and immutable objects
- Master Python's memory management with reference counting and garbage collection
- Know when to use `is` vs `==` for comparisons
- Understand shallow vs deep copying for complex data structures
- Choose the right data structure for performance optimization
- Leverage built-in collections (Counter, defaultdict, deque) for efficiency
- Understand string immutability and optimization techniques
- Master hashing concepts for sets and dictionaries
- Prefer list comprehensions for better performance and readability
- Understand object lifecycle and garbage collection mechanisms

---

**Next:** [Day 02](../Day-02/README.md)

