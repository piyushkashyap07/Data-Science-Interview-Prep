# 📊 SQL Joins & Window Functions - Day 4/50

---

## **1) What are the main differences between INNER, LEFT, RIGHT, and FULL OUTER JOINs?**

**Simple explanation:**
- **INNER JOIN**: Returns only matching rows from both tables
- **LEFT JOIN**: Returns all rows from left table + matching rows from right table
- **RIGHT JOIN**: Returns all rows from right table + matching rows from left table
- **FULL OUTER JOIN**: Returns all rows from both tables (matched + unmatched)

**Example:**

```sql
-- Sample tables
Employees: id | name
           1  | Alice
           2  | Bob
           3  | Charlie

Departments: emp_id | dept
             1      | HR
             2      | IT

-- INNER JOIN (only matching)
SELECT e.name, d.dept
FROM Employees e
INNER JOIN Departments d ON e.id = d.emp_id;
-- Result: Alice-HR, Bob-IT

-- LEFT JOIN (all from left + matches)
SELECT e.name, d.dept
FROM Employees e
LEFT JOIN Departments d ON e.id = d.emp_id;
-- Result: Alice-HR, Bob-IT, Charlie-NULL

-- RIGHT JOIN (all from right + matches)
SELECT e.name, d.dept
FROM Employees e
RIGHT JOIN Departments d ON e.id = d.emp_id;
-- Result: Alice-HR, Bob-IT

-- FULL OUTER JOIN (all from both)
SELECT e.name, d.dept
FROM Employees e
FULL OUTER JOIN Departments d ON e.id = d.emp_id;
-- Result: Alice-HR, Bob-IT, Charlie-NULL
```

---

## **2) How would you write a query to return rows that exist in one table but not in another?**

**Simple explanation:** Use `LEFT JOIN` with `WHERE NULL` or `NOT IN` / `NOT EXISTS`

**Example:**

```sql
-- Find employees who don't have a department assigned

-- Method 1: LEFT JOIN with NULL check
SELECT e.*
FROM Employees e
LEFT JOIN Departments d ON e.id = d.emp_id
WHERE d.emp_id IS NULL;

-- Method 2: NOT IN
SELECT *
FROM Employees
WHERE id NOT IN (SELECT emp_id FROM Departments);

-- Method 3: NOT EXISTS
SELECT *
FROM Employees e
WHERE NOT EXISTS (
    SELECT 1 FROM Departments d WHERE d.emp_id = e.id
);
```

---

## **3) What is a SELF JOIN, and in what situations is it useful?**

**Simple explanation:** A SELF JOIN is when a table is joined with itself. Useful for comparing rows within the same table or finding hierarchical relationships.

**Example:**

```sql
-- Employees table with manager info
Employees: id | name    | manager_id
           1  | Alice   | NULL
           2  | Bob     | 1
           3  | Charlie | 1
           4  | David   | 2

-- Find employee names with their manager names
SELECT 
    e.name AS employee,
    m.name AS manager
FROM Employees e
LEFT JOIN Employees m ON e.manager_id = m.id;

-- Result:
-- Alice    | NULL
-- Bob      | Alice
-- Charlie  | Alice
-- David    | Bob
```

**Other use cases:**
- Finding pairs of employees in the same department
- Comparing salaries between employees
- Finding employees hired on the same date

---

## **4) How can you fetch the second-highest salary from an Employee table?**

**Simple explanation:** Multiple approaches - use LIMIT with OFFSET, subquery, or DENSE_RANK()

**Example:**

```sql
-- Method 1: Using LIMIT and OFFSET
SELECT DISTINCT salary
FROM Employee
ORDER BY salary DESC
LIMIT 1 OFFSET 1;

-- Method 2: Using subquery
SELECT MAX(salary)
FROM Employee
WHERE salary < (SELECT MAX(salary) FROM Employee);

-- Method 3: Using DENSE_RANK() window function
SELECT salary
FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) as rank
    FROM Employee
) ranked
WHERE rank = 2;
```

---

## **5) What is the difference between PARTITION BY and GROUP BY in SQL?**

**Simple explanation:**
- **GROUP BY**: Collapses rows into groups, reduces number of rows
- **PARTITION BY**: Divides data into groups but keeps all rows, used with window functions

**Example:**

```sql
-- Sample data
Sales: dept  | employee | amount
       IT    | Alice    | 1000
       IT    | Bob      | 1500
       HR    | Charlie  | 800
       HR    | David    | 1200

-- GROUP BY - collapses to one row per dept
SELECT dept, SUM(amount) as total
FROM Sales
GROUP BY dept;
-- Result: IT-2500, HR-2000 (only 2 rows)

-- PARTITION BY - keeps all rows
SELECT 
    dept,
    employee,
    amount,
    SUM(amount) OVER (PARTITION BY dept) as dept_total
FROM Sales;
-- Result: All 4 rows with dept totals added
-- IT | Alice   | 1000 | 2500
-- IT | Bob     | 1500 | 2500
-- HR | Charlie | 800  | 2000
-- HR | David   | 1200 | 2000
```

---

## **6) How would you use ROW_NUMBER(), RANK(), or DENSE_RANK() to solve ranking problems?**

**Simple explanation:**
- **ROW_NUMBER()**: Sequential numbering (1,2,3,4...)
- **RANK()**: Same values get same rank, skips next numbers (1,2,2,4...)
- **DENSE_RANK()**: Same values get same rank, no gaps (1,2,2,3...)

**Example:**

```sql
-- Sample data
Scores: name    | score
        Alice   | 95
        Bob     | 90
        Charlie | 90
        David   | 85

SELECT 
    name,
    score,
    ROW_NUMBER() OVER (ORDER BY score DESC) as row_num,
    RANK() OVER (ORDER BY score DESC) as rank,
    DENSE_RANK() OVER (ORDER BY score DESC) as dense_rank
FROM Scores;

-- Result:
-- Alice   | 95 | 1 | 1 | 1
-- Bob     | 90 | 2 | 2 | 2
-- Charlie | 90 | 3 | 2 | 2  (note the difference!)
-- David   | 85 | 4 | 4 | 3  (RANK skips 3, DENSE_RANK doesn't)
```

---

## **7) How do you calculate a running total or cumulative sum with SQL?**

**Simple explanation:** Use `SUM()` with `OVER()` and `ORDER BY`

**Example:**

```sql
-- Sample data
Transactions: date       | amount
              2024-01-01 | 100
              2024-01-02 | 150
              2024-01-03 | 200
              2024-01-04 | 50

SELECT 
    date,
    amount,
    SUM(amount) OVER (ORDER BY date) as running_total
FROM Transactions;

-- Result:
-- 2024-01-01 | 100 | 100
-- 2024-01-02 | 150 | 250  (100+150)
-- 2024-01-03 | 200 | 450  (100+150+200)
-- 2024-01-04 | 50  | 500  (100+150+200+50)

-- Running total by department
SELECT 
    dept,
    date,
    amount,
    SUM(amount) OVER (PARTITION BY dept ORDER BY date) as dept_running_total
FROM Transactions;
```

---

## **8) What is the purpose of LAG() and LEAD() functions? Give an example.**

**Simple explanation:**
- **LAG()**: Access previous row's value
- **LEAD()**: Access next row's value

Great for comparing current row with previous/next rows!

**Example:**

```sql
-- Sample data
Sales: month | revenue
       Jan   | 10000
       Feb   | 12000
       Mar   | 11000
       Apr   | 15000

SELECT 
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month) as prev_month_revenue,
    LEAD(revenue) OVER (ORDER BY month) as next_month_revenue,
    revenue - LAG(revenue) OVER (ORDER BY month) as month_over_month_change
FROM Sales;

-- Result:
-- Jan | 10000 | NULL  | 12000 | NULL
-- Feb | 12000 | 10000 | 11000 | 2000  (growth)
-- Mar | 11000 | 12000 | 15000 | -1000 (decline)
-- Apr | 15000 | 11000 | NULL  | 4000  (growth)
```

**Real use case:** Calculate day-over-day user growth, stock price changes, etc.

---

## **9) How do you find the top 3 salaries in each department using window functions?**

**Simple explanation:** Use `DENSE_RANK()` or `ROW_NUMBER()` with `PARTITION BY department`

**Example:**

```sql
-- Sample data
Employees: dept | name    | salary
           IT   | Alice   | 90000
           IT   | Bob     | 85000
           IT   | Charlie | 80000
           IT   | David   | 75000
           HR   | Eve     | 70000
           HR   | Frank   | 68000
           HR   | Grace   | 65000

-- Solution
SELECT dept, name, salary
FROM (
    SELECT 
        dept,
        name,
        salary,
        DENSE_RANK() OVER (PARTITION BY dept ORDER BY salary DESC) as salary_rank
    FROM Employees
) ranked
WHERE salary_rank <= 3;

-- Result:
-- IT | Alice   | 90000
-- IT | Bob     | 85000
-- IT | Charlie | 80000
-- HR | Eve     | 70000
-- HR | Frank   | 68000
-- HR | Grace   | 65000
```

---

## **10) Why can't window functions be used directly in a WHERE clause, and what's the workaround?**

**Simple explanation:** SQL processes clauses in this order:
1. FROM/JOIN
2. WHERE
3. GROUP BY
4. **Window functions** (happen here)
5. SELECT
6. ORDER BY

Window functions are evaluated **after** WHERE, so you can't filter on them directly!

**Workaround:** Use a subquery or CTE

**Example:**

```sql
-- ❌ This WON'T work:
SELECT 
    name,
    salary,
    RANK() OVER (ORDER BY salary DESC) as rank
FROM Employees
WHERE rank <= 3;  -- ERROR! Can't use window function in WHERE

-- ✅ This WILL work - Method 1: Subquery
SELECT *
FROM (
    SELECT 
        name,
        salary,
        RANK() OVER (ORDER BY salary DESC) as rank
    FROM Employees
) ranked
WHERE rank <= 3;

-- ✅ This WILL work - Method 2: CTE (Common Table Expression)
WITH ranked_employees AS (
    SELECT 
        name,
        salary,
        RANK() OVER (ORDER BY salary DESC) as rank
    FROM Employees
)
SELECT *
FROM ranked_employees
WHERE rank <= 3;
```

---

## 🎯 **Quick Recap Cheat Sheet:**

**Joins:** INNER (both match), LEFT (all left), RIGHT (all right), FULL (all both)

**Window vs GROUP:** GROUP BY collapses rows, PARTITION BY keeps them

**Ranking:** ROW_NUMBER (1,2,3), RANK (1,2,2,4), DENSE_RANK (1,2,2,3)

**Window Syntax:** `FUNCTION() OVER (PARTITION BY col ORDER BY col)`

**Workaround for WHERE:** Use subquery or CTE

---

## 📝 **Key Takeaways**

### **Joins:**
- **INNER JOIN** returns only matching records from both tables
- **LEFT JOIN** preserves all records from the left table, adding NULLs for non-matches
- **RIGHT JOIN** preserves all records from the right table
- **FULL OUTER JOIN** preserves all records from both tables
- Use **LEFT JOIN + WHERE NULL** to find non-matching records
- **SELF JOIN** compares rows within the same table (hierarchies, pairs, comparisons)

### **Window Functions:**
- Window functions **don't collapse rows** like GROUP BY does
- Syntax: `FUNCTION() OVER (PARTITION BY col ORDER BY col)`
- **PARTITION BY** divides data into groups while keeping all rows visible
- Window functions are evaluated **after WHERE** clause, so use subqueries or CTEs to filter results

### **Ranking Functions:**
- **ROW_NUMBER()**: Unique sequential numbers (1, 2, 3, 4...)
- **RANK()**: Same rank for ties, skips next values (1, 2, 2, 4...)
- **DENSE_RANK()**: Same rank for ties, no gaps (1, 2, 2, 3...)
- Use with **PARTITION BY** to rank within groups (e.g., top 3 per department)

### **Analytical Functions:**
- **LAG()**: Access previous row's value (great for period-over-period analysis)
- **LEAD()**: Access next row's value
- **SUM() OVER (ORDER BY)**: Calculate running totals and cumulative sums
- **AVG() OVER()**: Calculate moving averages with window frames

### **Best Practices:**
- Use **CTEs** (WITH clause) for better query readability
- Prefer **NOT EXISTS** over **NOT IN** for better NULL handling
- Use **DENSE_RANK()** for Nth highest problems to handle duplicates
- Always specify **ORDER BY** in window functions when order matters
- Test join types with small datasets first to verify expected results

### **Common Patterns:**
- **Second highest value**: Use DENSE_RANK() or LIMIT/OFFSET
- **Running totals**: SUM() OVER (ORDER BY date)
- **Month-over-month change**: Current - LAG(value)
- **Top N per group**: RANK() with PARTITION BY + WHERE in subquery
- **Find orphan records**: LEFT JOIN + WHERE IS NULL

---

**Previous:** [Day 03](../Day-03-Pandas/README.md) | **Next:** [Day 05](../Day-05/README.md)
