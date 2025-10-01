# Day 02 - NumPy Fundamentals: Comprehensive Guide

**Topics Covered:** NumPy Arrays, Broadcasting, Vectorization, Performance Optimization

---

## Question 1: NumPy Array vs Python List

**Topic:** Python/NumPy  
**Difficulty:** Intermediate

### Question
How is a NumPy array different from a regular Python list?

### Answer

**Key Differences:**
- **Type homogeneity:** NumPy arrays store elements of the same data type, while Python lists can store mixed types
- **Memory efficiency:** NumPy arrays use contiguous memory blocks, lists store references to objects scattered in memory
- **Operations:** NumPy supports vectorized operations, lists require loops
- **Dimensions:** NumPy arrays can be multidimensional naturally, lists need nesting

```python
import numpy as np

# Python list - mixed types allowed
python_list = [1, 2.5, "hello", True]
print(python_list)  # [1, 2.5, 'hello', True]

# NumPy array - homogeneous type
numpy_array = np.array([1, 2, 3, 4])
print(numpy_array)  # [1 2 3 4]
print(numpy_array.dtype)  # int64

# Try mixed types in NumPy - it converts to common type
mixed = np.array([1, 2.5, 3])
print(mixed)  # [1.  2.5 3. ]
print(mixed.dtype)  # float64
```

---

## Question 2: NumPy Performance

**Topic:** Python/NumPy  
**Difficulty:** Intermediate

### Question
Why does NumPy usually run much faster than lists in Python?

### Answer

**Reasons for speed:**
1. **Pre-compiled C code:** NumPy operations are implemented in C/Fortran
2. **Contiguous memory:** Data stored sequentially allows CPU cache optimization
3. **Vectorization:** Operations applied to entire arrays without Python loops
4. **No type checking:** Since all elements are the same type, no runtime type checking needed

```python
import numpy as np
import time

# Create large dataset
size = 1000000

# Using Python list
python_list = list(range(size))
start = time.time()
result_list = [x * 2 for x in python_list]
list_time = time.time() - start

# Using NumPy array
numpy_array = np.arange(size)
start = time.time()
result_numpy = numpy_array * 2
numpy_time = time.time() - start

print(f"List time: {list_time:.4f}s")
print(f"NumPy time: {numpy_time:.4f}s")
print(f"NumPy is {list_time/numpy_time:.1f}x faster")
# NumPy is typically 10-100x faster!
```

---

## Question 3: Broadcasting in NumPy

**Topic:** Python/NumPy  
**Difficulty:** Intermediate

### Question
Can you explain broadcasting in NumPy with a simple example?

### Answer

Broadcasting allows NumPy to perform operations on arrays of different shapes by automatically expanding smaller arrays to match larger ones (without copying data).

```python
import numpy as np

# Example 1: Scalar broadcasting
arr = np.array([1, 2, 3, 4])
result = arr + 10
print(result)  # [11 12 13 14]
# 10 is "broadcast" to [10, 10, 10, 10]

# Example 2: 1D array with 2D array
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

row = np.array([10, 20, 30])
result = matrix + row
print(result)
# [[11 22 33]
#  [14 25 36]
#  [17 28 39]]
# row is broadcast to each row of the matrix

# Example 3: Column broadcasting
column = np.array([[10],
                   [20],
                   [30]])
result = matrix + column
print(result)
# [[11 12 13]
#  [24 25 26]
#  [37 38 39]]
```

---

## Question 4: Array Attributes - shape, size, ndim

**Topic:** Python/NumPy  
**Difficulty:** Intermediate

### Question
What's the difference between `.shape`, `.size`, and `.ndim` in an ndarray?

### Answer

```python
import numpy as np

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# .shape - tuple showing dimensions (rows, columns, ...)
print(f"Shape: {arr.shape}")  # (3, 4) - 3 rows, 4 columns

# .size - total number of elements
print(f"Size: {arr.size}")  # 12 (3 × 4)

# .ndim - number of dimensions/axes
print(f"Dimensions: {arr.ndim}")  # 2 (2D array)

# Another example with 3D array
arr_3d = np.array([[[1, 2], [3, 4]], 
                   [[5, 6], [7, 8]]])
print(f"3D Shape: {arr_3d.shape}")  # (2, 2, 2)
print(f"3D Size: {arr_3d.size}")    # 8
print(f"3D Ndim: {arr_3d.ndim}")    # 3
```

**Key Points:**
- `.shape` returns tuple of dimensions
- `.size` returns total element count
- `.ndim` returns number of axes/dimensions

---

## Question 5: Reshaping Arrays

**Topic:** Python/NumPy  
**Difficulty:** Intermediate

### Question
How do you reshape a 1D array into 2D in NumPy?

### Answer

```python
import numpy as np

# Create a 1D array
arr_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
print(f"Original: {arr_1d.shape}")  # (12,)

# Reshape to 2D - different configurations
arr_3x4 = arr_1d.reshape(3, 4)
print(f"3x4 array:\n{arr_3x4}")
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]

arr_4x3 = arr_1d.reshape(4, 3)
print(f"4x3 array:\n{arr_4x3}")

# Use -1 to auto-calculate one dimension
arr_auto = arr_1d.reshape(2, -1)  # -1 means "figure this out"
print(f"2x6 array:\n{arr_auto}")  # NumPy calculates 6 automatically
print(f"Shape: {arr_auto.shape}")  # (2, 6)

# Reshape to column vector
column = arr_1d.reshape(-1, 1)
print(f"Column shape: {column.shape}")  # (12, 1)
```

**Important:** Total elements must remain the same (3×4 = 4×3 = 2×6 = 12)

---

## Question 6: View vs Copy

**Topic:** Python/NumPy  
**Difficulty:** Intermediate

### Question
What's the difference between `view()` and `copy()` methods?

### Answer

View creates a new array object that looks at the same data, while copy creates a completely independent array with duplicated data.

```python
import numpy as np

original = np.array([1, 2, 3, 4, 5])

# Creating a VIEW - shares memory
view_arr = original.view()
view_arr[0] = 999
print(f"Original after view change: {original}")  # [999 2 3 4 5]
print(f"View: {view_arr}")  # [999 2 3 4 5]
# Both changed!

# Reset
original = np.array([1, 2, 3, 4, 5])

# Creating a COPY - independent memory
copy_arr = original.copy()
copy_arr[0] = 999
print(f"Original after copy change: {original}")  # [1 2 3 4 5]
print(f"Copy: {copy_arr}")  # [999 2 3 4 5]
# Only copy changed!

# Slicing creates a view by default
original = np.array([1, 2, 3, 4, 5])
slice_arr = original[1:4]
slice_arr[0] = 777
print(f"Original after slice change: {original}")  # [1 777 3 4 5]
# Slicing creates a view, not a copy!
```

**Key Insight:** Slicing creates views by default. Use `.copy()` when you need independent data.

---

## Question 7: Axis Parameter

**Topic:** Python/NumPy  
**Difficulty:** Intermediate

### Question
Why do many NumPy functions have an `axis` argument? Give an example.

### Answer

The axis argument specifies which dimension to perform the operation along. This is crucial for multidimensional arrays.

- **axis=0:** operate down the rows (column-wise)
- **axis=1:** operate across the columns (row-wise)
- **axis=None:** operate on flattened array (default for many functions)

```python
import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Sum examples
print(f"Total sum: {np.sum(arr)}")  # 45 (all elements)
print(f"Sum axis=0: {np.sum(arr, axis=0)}")  # [12 15 18] (sum each column)
print(f"Sum axis=1: {np.sum(arr, axis=1)}")  # [ 6 15 24] (sum each row)

# Mean examples
print(f"Mean axis=0: {np.mean(arr, axis=0)}")  # [4. 5. 6.] (avg of each column)
print(f"Mean axis=1: {np.mean(arr, axis=1)}")  # [2. 5. 8.] (avg of each row)

# Max examples
print(f"Max axis=0: {np.max(arr, axis=0)}")  # [7 8 9] (max of each column)
print(f"Max axis=1: {np.max(arr, axis=1)}")  # [3 6 9] (max of each row)

# Visual explanation:
# axis=0 (↓)  axis=1 (→)
# [[1, 2, 3],
#  [4, 5, 6],
#  [7, 8, 9]]
```

---

## Question 8: Handling Missing Values (NaN)

**Topic:** Python/NumPy  
**Difficulty:** Intermediate

### Question
How do you deal with NaN or missing values in a NumPy array?

### Answer

```python
import numpy as np

# Create array with NaN values
arr = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])
print(f"Original: {arr}")

# Check for NaN values
print(f"Is NaN: {np.isnan(arr)}")  # [False False True False True False]
print(f"Count of NaN: {np.sum(np.isnan(arr))}")  # 2

# Remove NaN values
clean_arr = arr[~np.isnan(arr)]  # ~ means NOT
print(f"Without NaN: {clean_arr}")  # [1. 2. 4. 6.]

# Replace NaN with a value
arr_filled = np.where(np.isnan(arr), 0, arr)
print(f"NaN replaced with 0: {arr_filled}")  # [1. 2. 0. 4. 0. 6.]

# Use nanmean, nansum, etc. to ignore NaN
print(f"Mean (ignoring NaN): {np.nanmean(arr)}")  # 3.25
print(f"Sum (ignoring NaN): {np.nansum(arr)}")    # 13.0

# Fill NaN with mean
mean_value = np.nanmean(arr)
arr_mean_filled = np.where(np.isnan(arr), mean_value, arr)
print(f"NaN filled with mean: {arr_mean_filled}")

# 2D array example
matrix = np.array([[1, 2, np.nan],
                   [4, np.nan, 6],
                   [7, 8, 9]])
print(f"Column means (ignore NaN): {np.nanmean(matrix, axis=0)}")
```

---

## Question 9: Vectorization Advantages

**Topic:** Python/NumPy  
**Difficulty:** Intermediate

### Question
What's the advantage of vectorization compared to writing loops in NumPy?

### Answer

**Vectorization advantages:**
1. Much faster execution (10-100x)
2. More readable and concise code
3. Less prone to bugs
4. Automatically optimized by NumPy's C implementation

```python
import numpy as np
import time

# Create sample data
size = 1000000
x = np.random.rand(size)
y = np.random.rand(size)

# METHOD 1: Loop (slow)
start = time.time()
result_loop = np.zeros(size)
for i in range(size):
    result_loop[i] = x[i] * y[i] + x[i]
loop_time = time.time() - start

# METHOD 2: Vectorized (fast)
start = time.time()
result_vectorized = x * y + x
vectorized_time = time.time() - start

print(f"Loop time: {loop_time:.4f}s")
print(f"Vectorized time: {vectorized_time:.4f}s")
print(f"Speedup: {loop_time/vectorized_time:.1f}x")
print(f"Results equal: {np.allclose(result_loop, result_vectorized)}")

# More complex example
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Loop version
result_loop = []
for x in arr:
    if x % 2 == 0:
        result_loop.append(x ** 2)
    else:
        result_loop.append(x * 3)

# Vectorized version (much cleaner!)
result_vec = np.where(arr % 2 == 0, arr ** 2, arr * 3)

print(f"Loop result: {result_loop}")
print(f"Vectorized result: {result_vec}")
```

---

## Question 10: Random Number Generation and Seeds

**Topic:** Python/NumPy  
**Difficulty:** Intermediate

### Question
How can you generate random numbers using NumPy, and why do we set a random seed?

### Answer

Setting a random seed ensures **reproducibility** - you get the same "random" numbers each time, which is crucial for debugging, testing, and scientific reproducibility.

```python
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate random numbers - various methods
print("Random float [0, 1):", np.random.rand(5))
# [0.37454012 0.95071431 0.73199394 0.59865848 0.15601864]

# Random integers
print("Random int [0, 10):", np.random.randint(0, 10, size=5))
# [6 3 7 4 6]

# Random floats in range
print("Random [10, 20):", np.random.uniform(10, 20, size=5))
# [17.20 11.82 13.95 19.53 16.84]

# Normal distribution (mean=0, std=1)
print("Normal distribution:", np.random.randn(5))
# [-0.10321885  0.41059850  0.14404357  1.45427351  0.76103773]

# Custom normal distribution
print("Normal (mean=100, std=15):", np.random.normal(100, 15, size=5))
# [103.92  95.67 108.45  98.23 112.34]

# Random choice from array
colors = np.array(['red', 'blue', 'green', 'yellow'])
print("Random choice:", np.random.choice(colors, size=3))
# ['blue' 'red' 'blue']

# Shuffle an array
arr = np.array([1, 2, 3, 4, 5])
np.random.shuffle(arr)
print("Shuffled:", arr)

# WHY USE SEED?
print("\n--- Demonstrating seed importance ---")

# With seed - reproducible
np.random.seed(42)
print("Run 1:", np.random.rand(3))
np.random.seed(42)
print("Run 2:", np.random.rand(3))  # Same numbers!

# Without seed - different each time
print("No seed 1:", np.random.rand(3))
print("No seed 2:", np.random.rand(3))  # Different numbers!

# Modern way (recommended for new code)
rng = np.random.default_rng(seed=42)
print("Modern RNG:", rng.random(5))
```

---

## 📚 Additional Resources

- [NumPy Official Documentation](https://numpy.org/doc/stable/)
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [NumPy for Beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [Broadcasting Rules Explained](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [From Python to NumPy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/)

---

## 🎯 Key Takeaways

- NumPy arrays are homogeneous, memory-efficient, and support vectorized operations
- Broadcasting enables operations on different-shaped arrays without explicit loops
- Understanding `.shape`, `.size`, and `.ndim` is crucial for array manipulation
- Use `.copy()` for independent data, slicing creates views by default
- The `axis` parameter controls operation direction in multidimensional arrays
- Handle NaN values with `np.isnan()`, `np.nanmean()`, and related functions
- Vectorization is 10-100x faster than Python loops
- Always set random seeds for reproducible results in experiments
- Views share memory with original arrays, copies don't
- Use `-1` in reshape to auto-calculate dimensions

---

**Previous:** [Day 01 - Python](../Day-01-Python/README.md) | **Next:** [Day 03 - Pandas](../Day-03-Pandas/README.md)
