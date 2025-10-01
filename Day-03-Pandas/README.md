# Day 03 - Pandas Interview Questions: Comprehensive Guide

**Topics Covered:** Pandas DataFrames, Performance Optimization, Data Manipulation, Time Series

---

## Question 1: Vectorization vs Apply/Lambda

**Topic:** Python/Pandas  
**Difficulty:** Intermediate

### Question
How can performance be improved using vectorization compared to apply or lambda in Pandas?

### Answer

Vectorization uses optimized C/Cython code under the hood and operates on entire arrays at once, making it 10-100x faster than `apply()` or lambda functions which loop through rows in Python.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'A': range(1000000), 'B': range(1000000)})

# Slow: Using apply with lambda
%timeit df['C'] = df.apply(lambda row: row['A'] + row['B'], axis=1)
# ~1-2 seconds

# Fast: Vectorized operation
%timeit df['C'] = df['A'] + df['B']
# ~10 milliseconds

# Another example: conditional logic
# Slow way
df['Category'] = df['A'].apply(lambda x: 'High' if x > 500000 else 'Low')

# Fast way (vectorized)
df['Category'] = np.where(df['A'] > 500000, 'High', 'Low')
```

**Key Points:**
- Vectorization is 10-100x faster
- Use NumPy operations and built-in Pandas methods
- Avoid `apply()` with `axis=1` when possible
- Use `np.where()` for conditional logic

---

## Question 2: Pandas Indexing Methods

**Topic:** Python/Pandas  
**Difficulty:** Intermediate

### Question
What are the differences between `.at[]`, `.iat[]`, `.loc[]`, and `.iloc[]`?

### Answer

These are accessor methods with different use cases based on whether you're using labels or positions, and whether accessing single or multiple values.

```python
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['NYC', 'LA', 'Chicago']
}, index=['A', 'B', 'C'])

# .loc[] - Label-based indexing (rows and columns by name)
print(df.loc['A', 'Name'])  # 'Alice'
print(df.loc['A':'B', ['Name', 'Age']])  # Slicing with labels (inclusive)

# .iloc[] - Integer position-based indexing
print(df.iloc[0, 0])  # 'Alice'
print(df.iloc[0:2, 0:2])  # Slicing with positions (exclusive end)

# .at[] - Fast scalar access using labels (single value only)
print(df.at['A', 'Name'])  # 'Alice' - faster than .loc for single values
df.at['A', 'Age'] = 26  # Fast assignment

# .iat[] - Fast scalar access using integer positions
print(df.iat[0, 0])  # 'Alice' - faster than .iloc for single values
df.iat[0, 1] = 26  # Fast assignment
```

**Key Differences:**
- `.loc[]` and `.iloc[]`: Can select multiple rows/columns
- `.at[]` and `.iat[]`: Single value only, but ~2x faster
- `loc/at` use labels, `iloc/iat` use integer positions

---

## Question 3: Multi-Indexing in Pandas

**Topic:** Python/Pandas  
**Difficulty:** Intermediate

### Question
When and why would you use multi-indexing in Pandas?

### Answer

Multi-indexing creates multiple levels of indices, useful for representing higher-dimensional data in 2D structure. Use it for time series with multiple categories, grouped data with natural hierarchy, or panel data.

```python
# Creating multi-index DataFrame
arrays = [
    ['2024', '2024', '2024', '2025', '2025', '2025'],
    ['Q1', 'Q2', 'Q3', 'Q1', 'Q2', 'Q3']
]
index = pd.MultiIndex.from_arrays(arrays, names=['Year', 'Quarter'])

df = pd.DataFrame({
    'Revenue': [100, 120, 130, 140, 150, 160],
    'Profit': [20, 25, 28, 30, 32, 35]
}, index=index)

print(df)
#               Revenue  Profit
# Year Quarter                 
# 2024 Q1           100      20
#      Q2           120      25
#      Q3           130      28
# 2025 Q1           140      30
#      Q2           150      32
#      Q3           160      35

# Access specific level
print(df.loc['2024'])  # All Q1-Q3 for 2024
print(df.loc[('2024', 'Q1')])  # Specific quarter

# Cross-section
print(df.xs('Q1', level='Quarter'))  # All Q1 data across years

# Aggregation by level
print(df.groupby(level='Year').sum())
```

**Use Cases:**
- Hierarchical/grouped data
- Time series with categories
- Panel data (multiple dimensions)
- Easy aggregation at different levels

---

## Question 4: Time Series Handling

**Topic:** Python/Pandas  
**Difficulty:** Intermediate

### Question
How is time series data typically handled in Pandas?

### Answer

Pandas has rich functionality for time series through DatetimeIndex and specialized methods for resampling, shifting, rolling windows, and date-based slicing.

```python
# Creating time series
dates = pd.date_range('2024-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(100), index=dates)

# Parsing dates
df = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    'value': [10, 20, 30]
})
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Resampling (aggregating to different frequency)
monthly = ts.resample('M').mean()  # Monthly average
weekly_sum = ts.resample('W').sum()  # Weekly sum

# Shifting and lagging
df['previous_day'] = ts.shift(1)  # Lag by 1 day
df['next_day'] = ts.shift(-1)  # Lead by 1 day

# Rolling windows
df['rolling_7day_avg'] = ts.rolling(window=7).mean()
df['rolling_std'] = ts.rolling(window=7).std()

# Date components
df['year'] = ts.index.year
df['month'] = ts.index.month
df['day_of_week'] = ts.index.dayofweek

# Time-based slicing
print(ts['2024-01'])  # All January 2024 data
print(ts['2024-01-15':'2024-01-20'])  # Date range
```

**Key Features:**
- DatetimeIndex for time-based operations
- Resampling for frequency conversion
- Rolling windows for moving averages
- Easy date-based slicing

---

## Question 5: Stack vs Unstack

**Topic:** Python/Pandas  
**Difficulty:** Intermediate

### Question
What's the difference between `stack()` and `unstack()` methods?

### Answer

`stack()` moves columns to row index (wide to long format), while `unstack()` moves row index to columns (long to wide format). They're inverse operations for reshaping DataFrames.

```python
# Creating sample DataFrame
df = pd.DataFrame({
    'Product': ['A', 'A', 'B', 'B'],
    'Region': ['East', 'West', 'East', 'West'],
    'Sales': [100, 150, 200, 250]
})

# Pivot to multi-index
df_pivot = df.set_index(['Product', 'Region'])
print(df_pivot)
#                 Sales
# Product Region       
# A       East      100
#         West      150
# B       East      200
#         West      250

# unstack() - Move inner index level to columns (wide format)
df_wide = df_pivot.unstack()
print(df_wide)
#         Sales     
# Region   East West
# Product           
# A         100  150
# B         200  250

# stack() - Move columns to inner index level (long format)
df_long = df_wide.stack()
print(df_long)
# Back to original multi-index format

# Practical use case: preparing data for visualization
df_plot = df.pivot(index='Product', columns='Region', values='Sales')
# Easy to plot with df_plot.plot(kind='bar')
```

**Key Points:**
- `unstack()`: Long → Wide format (columns expand)
- `stack()`: Wide → Long format (rows expand)
- Inverse operations of each other
- Useful for data reshaping and visualization prep

---

## Question 6: Detecting and Dealing with Outliers

**Topic:** Python/Pandas  
**Difficulty:** Intermediate

### Question
How do you detect and deal with outliers using Pandas?

### Answer

Common methods include Z-score, IQR (Interquartile Range), and percentile-based detection. Dealing with outliers involves removal, capping, replacement, or transformation.

```python
df = pd.DataFrame({
    'value': [10, 12, 13, 11, 100, 14, 15, 13, 200, 12]
})

# Method 1: Z-score (statistical approach)
from scipy import stats
df['z_score'] = np.abs(stats.zscore(df['value']))
outliers_z = df[df['z_score'] > 3]

# Method 2: IQR (Interquartile Range) - most common
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_iqr = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
print(f"Outliers: {outliers_iqr['value'].tolist()}")

# Method 3: Percentile-based
lower_percentile = df['value'].quantile(0.05)
upper_percentile = df['value'].quantile(0.95)
outliers_percentile = df[(df['value'] < lower_percentile) | 
                          (df['value'] > upper_percentile)]

# Dealing with outliers:

# Option 1: Remove
df_clean = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]

# Option 2: Cap (winsorize)
df['value_capped'] = df['value'].clip(lower=lower_bound, upper=upper_bound)

# Option 3: Replace with median
median_val = df['value'].median()
df['value_replaced'] = df['value'].apply(
    lambda x: median_val if x > upper_bound or x < lower_bound else x
)

# Option 4: Log transformation (for right-skewed data)
df['value_log'] = np.log1p(df['value'])  # log(1+x) to handle zeros
```

**Detection Methods:**
- Z-score: Statistical (assumes normal distribution)
- IQR: Robust, works with any distribution
- Percentile: Simple, intuitive

**Handling Strategies:**
- Remove: Simplest but loses data
- Cap: Preserves data points
- Replace: Use median/mean
- Transform: Log, Box-Cox

---

## Question 7: Query Method vs Boolean Indexing

**Topic:** Python/Pandas  
**Difficulty:** Intermediate

### Question
How does the `.query()` method differ from boolean indexing?

### Answer

Both filter DataFrames, but `.query()` offers cleaner syntax for complex conditions, uses string expressions, and can leverage numexpr for better performance on large datasets.

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 40],
    'salary': [50000, 60000, 70000, 80000],
    'department': ['HR', 'IT', 'IT', 'HR']
})

# Boolean indexing (traditional)
result1 = df[(df['age'] > 25) & (df['salary'] < 75000)]

# .query() method (cleaner, more readable)
result2 = df.query('age > 25 and salary < 75000')

# Complex conditions with .query()
result3 = df.query('age > 25 and (department == "IT" or salary > 65000)')

# Using variables in .query()
min_age = 30
dept = 'IT'
result4 = df.query('age > @min_age and department == @dept')

# .query() with string methods
result5 = df.query('name.str.startswith("A")', engine='python')
```

**Advantages of .query():**
- More readable for complex conditions
- Can use `and`/`or` instead of `&`/`|`
- Easier to use variables with `@` syntax
- Sometimes faster for large DataFrames
- SQL-like syntax

---

## Question 8: Merging on Multiple Keys

**Topic:** Python/Pandas  
**Difficulty:** Intermediate

### Question
How can you merge or join DataFrames on multiple keys?

### Answer

Use `merge()` with a list of column names in the 'on' parameter, or use `join()` with multi-level indices. You can specify different join types (inner, outer, left, right).

```python
# Sample DataFrames
df1 = pd.DataFrame({
    'year': [2024, 2024, 2025, 2025],
    'quarter': ['Q1', 'Q2', 'Q1', 'Q2'],
    'region': ['East', 'East', 'West', 'West'],
    'sales': [100, 120, 130, 140]
})

df2 = pd.DataFrame({
    'year': [2024, 2024, 2025, 2025],
    'quarter': ['Q1', 'Q2', 'Q1', 'Q2'],
    'region': ['East', 'East', 'West', 'West'],
    'costs': [80, 90, 100, 110]
})

# Method 1: merge() with multiple keys
merged = pd.merge(df1, df2, on=['year', 'quarter', 'region'], how='inner')
print(merged)

# Method 2: Different key names
df3 = pd.DataFrame({
    'yr': [2024, 2024],
    'qtr': ['Q1', 'Q2'],
    'rgn': ['East', 'East'],
    'target': [110, 130]
})

merged2 = pd.merge(
    df1, df3,
    left_on=['year', 'quarter', 'region'],
    right_on=['yr', 'qtr', 'rgn'],
    how='left'
)

# Method 3: join() with multi-index
df1_indexed = df1.set_index(['year', 'quarter', 'region'])
df2_indexed = df2.set_index(['year', 'quarter', 'region'])
joined = df1_indexed.join(df2_indexed)

# Different join types
inner_join = pd.merge(df1, df2, on=['year', 'quarter'], how='inner')
outer_join = pd.merge(df1, df2, on=['year', 'quarter'], how='outer')
left_join = pd.merge(df1, df2, on=['year', 'quarter'], how='left')
right_join = pd.merge(df1, df2, on=['year', 'quarter'], how='right')
```

**Join Types:**
- **Inner:** Only matching rows
- **Outer:** All rows from both DataFrames
- **Left:** All rows from left DataFrame
- **Right:** All rows from right DataFrame

---

## Question 9: Categorical vs Object Data Type

**Topic:** Python/Pandas  
**Difficulty:** Intermediate

### Question
What is the difference between the categorical data type and object type in Pandas?

### Answer

Categorical type is optimized for storing repeated string values with significantly lower memory usage and faster operations. Object type is more flexible but less efficient for repeated values.

```python
# Creating sample data
df = pd.DataFrame({
    'id': range(1000000),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], 1000000)
})

# Object type (default for strings)
print(f"Object type memory: {df['city'].memory_usage(deep=True) / 1024**2:.2f} MB")
# ~60+ MB

# Convert to categorical
df['city_cat'] = df['city'].astype('category')
print(f"Categorical memory: {df['city_cat'].memory_usage(deep=True) / 1024**2:.2f} MB")
# ~1 MB (60x reduction!)

# Categorical benefits
print(df['city_cat'].cat.categories)  # ['Chicago', 'Houston', 'LA', 'NYC']
print(df['city_cat'].cat.codes)  # Integer codes: 0, 1, 2, 3

# Ordered categories
df['size'] = pd.Categorical(
    ['small', 'medium', 'large', 'medium', 'small'],
    categories=['small', 'medium', 'large'],
    ordered=True
)

# Now comparisons work
print(df['size'] > 'small')  # Returns boolean for medium and large
```

**Key Differences:**
- **Memory:** Categorical uses far less memory for repeated values
- **Performance:** Faster comparisons and groupby operations
- **Ordering:** Categorical supports ordered comparisons
- **Flexibility:** Object type more flexible for unique strings

**When to Use Categorical:**
- Repeated string values
- Limited set of unique values
- Memory is a concern
- Need ordered comparisons

---

## Question 10: Memory Optimization Techniques

**Topic:** Python/Pandas  
**Difficulty:** Intermediate

### Question
What techniques can you use to optimize memory usage in very large DataFrames?

### Answer

Key techniques include downcasting numeric types, converting to categorical, reading in chunks, selecting only needed columns, using sparse data structures, and storing in efficient formats like Parquet.

```python
# Sample large DataFrame
df = pd.DataFrame({
    'id': range(1000000),
    'value': np.random.randn(1000000),
    'category': np.random.choice(['A', 'B', 'C'], 1000000),
    'date': pd.date_range('2020-01-01', periods=1000000, freq='1min')
})

# Check initial memory
print(f"Initial memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Technique 1: Downcast numeric types
df['id'] = pd.to_numeric(df['id'], downcast='unsigned')
df['value'] = pd.to_numeric(df['value'], downcast='float')

# Technique 2: Convert to categorical
df['category'] = df['category'].astype('category')

# Technique 3: Read CSV with optimized dtypes
dtype_dict = {
    'id': 'uint32',
    'value': 'float32',
    'category': 'category'
}

# Technique 4: Read in chunks
chunk_size = 100000
chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size, dtype=dtype_dict):
    chunks.append(chunk)
df_optimized = pd.concat(chunks, ignore_index=True)

# Technique 5: Automatic optimization function
def optimize_dataframe(df):
    """Automatically optimize DataFrame memory usage"""
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        
        else:  # Object type
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')
    
    return df

df_optimized = optimize_dataframe(df)

# Technique 6: Use Parquet format
df.to_parquet('data.parquet', compression='gzip')
df_loaded = pd.read_parquet('data.parquet')
```

**Memory Reduction Summary:**
- Downcast numeric types: 30-50% reduction
- Categorical for strings: 50-90% reduction
- Combined optimization: Often 70-80% total reduction

**Best Practices:**
- Use appropriate data types from the start
- Convert repeated strings to categorical
- Read only needed columns
- Use chunking for extremely large files
- Store in Parquet format for efficiency

---

## 📚 Additional Resources

- [Pandas Official Documentation](https://pandas.pydata.org/docs/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Pandas Performance Tips](https://pandas.pydata.org/docs/user_guide/enhancingperf.html)
- [Modern Pandas](https://tomaugspurger.github.io/modern-1-intro.html)

---

## 🎯 Key Takeaways

- **Vectorization** is 10-100x faster than apply/lambda - always prefer it
- Use `.at[]`/`.iat[]` for single values, `.loc[]`/`.iloc[]` for multiple
- **Multi-indexing** enables hierarchical data representation in 2D
- Pandas has rich **time series** functionality with resampling and rolling windows
- `stack()`/`unstack()` are inverse operations for data reshaping
- **IQR method** is the most robust for outlier detection
- `.query()` provides cleaner, more readable filtering syntax
- Merge on **multiple keys** for complex join operations
- **Categorical dtype** saves massive memory for repeated strings
- Proper **memory optimization** can reduce DataFrame size by 70-80%

---

**Previous:** [Day 02 - NumPy](../Day-02-NumPy/README.md) | **Next:** [Day 04](../Day-04/README.md)
