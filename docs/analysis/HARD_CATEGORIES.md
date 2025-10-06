# 🔥 HARD PROBLEM CATEGORIES (What Linters Can't Do)

These are the categories Nerion should have focused on from day one.
Each requires **semantic understanding** that rule-based tools struggle with.

---

## 🎯 Tier 1: Dataflow & Control Flow (HIGHEST PRIORITY)

### 1. **Race Conditions & Data Races**
**Why Hard**: Requires modeling thread interleavings
```python
# Before (bug):
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        temp = self.count
        time.sleep(0.001)  # Simulation of work
        self.count = temp + 1  # Race! Other thread might increment between read and write

# After (fixed):
class Counter:
    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:
            self.count += 1
```

**Pylint Coverage**: ❌ None (can't model thread interleavings)
**GNN Potential**: ✅ High (can learn patterns of shared state + concurrent access)

---

### 2. **Deadlock Patterns**
**Why Hard**: Requires analyzing lock acquisition order across multiple functions
```python
# Before (bug):
def transfer_money(from_account, to_account, amount):
    from_account.lock.acquire()
    to_account.lock.acquire()  # Deadlock if another thread does opposite order
    from_account.balance -= amount
    to_account.balance += amount
    to_account.lock.release()
    from_account.lock.release()

# After (fixed):
def transfer_money(from_account, to_account, amount):
    # Always acquire locks in consistent order (by account ID)
    first, second = sorted([from_account, to_account], key=lambda a: a.id)
    first.lock.acquire()
    second.lock.acquire()
    from_account.balance -= amount
    to_account.balance += amount
    second.lock.release()
    first.lock.release()
```

**Pylint Coverage**: ❌ None
**GNN Potential**: ✅ High (learn lock ordering patterns)

---

### 3. **Tainted Data Flow (Security)**
**Why Hard**: Requires tracking data from source (user input) to sink (SQL/command)
```python
# Before (SQL injection):
def get_user(user_id):
    user_id = request.args.get('id')  # Tainted source
    # ... 50 lines of code ...
    query = f"SELECT * FROM users WHERE id = {user_id}"  # Tainted sink!
    return db.execute(query)

# After (fixed):
def get_user(user_id):
    user_id = request.args.get('id')
    # Sanitize before use
    query = "SELECT * FROM users WHERE id = ?"
    return db.execute(query, (int(user_id),))
```

**Pylint Coverage**: ❌ Limited (bandit catches f-strings in SQL, but not dataflow)
**GNN Potential**: ✅ Very High (GNN edges model dataflow perfectly)

---

### 4. **Resource Leaks with Exception Paths**
**Why Hard**: Requires analyzing all exception paths through function
```python
# Before (leak):
def process_file(filename):
    file = open(filename)  # Acquired
    data = file.read()
    result = risky_operation(data)  # Might throw exception
    file.close()  # Never reached if exception thrown
    return result

# After (fixed):
def process_file(filename):
    with open(filename) as file:  # Context manager ensures cleanup
        data = file.read()
        return risky_operation(data)
```

**Pylint Coverage**: ⚠️ Partial (warns about open() without context manager, but not dataflow)
**GNN Potential**: ✅ High (learn acquire → release patterns with exception edges)

---

### 5. **Use-After-Free (Python GC Edge Cases)**
**Why Hard**: Requires tracking object lifetime across async boundaries
```python
# Before (bug):
async def broken():
    buffer = bytearray(1024)
    asyncio.create_task(process_async(buffer))  # Task holds reference
    del buffer  # Might be freed before task completes
    return "done"

# After (fixed):
async def fixed():
    buffer = bytearray(1024)
    task = asyncio.create_task(process_async(buffer))
    await task  # Ensure task completes before buffer can be freed
    return "done"
```

**Pylint Coverage**: ❌ None
**GNN Potential**: ✅ Medium (async patterns are complex)

---

## 🎯 Tier 2: Framework-Specific Anti-Patterns

### 6. **Django N+1 Queries**
**Why Hard**: Requires understanding Django ORM semantics
```python
# Before (N+1 queries):
def get_posts_with_authors():
    posts = Post.objects.all()  # 1 query
    return [(p.title, p.author.name) for p in posts]  # N queries (1 per post)

# After (optimized):
def get_posts_with_authors():
    posts = Post.objects.select_related('author').all()  # 1 query with JOIN
    return [(p.title, p.author.name) for p in posts]
```

**Pylint Coverage**: ❌ None (requires Django ORM knowledge)
**GNN Potential**: ✅ Very High (common pattern in Django codebases)

---

### 7. **React useState Closure Stale State**
**Why Hard**: Requires understanding JavaScript closures (if extending to JS)
```javascript
// Before (stale state):
function Counter() {
  const [count, setCount] = useState(0);

  const increment = () => {
    setTimeout(() => {
      setCount(count + 1);  // Bug: Captures stale 'count' value
    }, 1000);
  };

  return <button onClick={increment}>Count: {count}</button>;
}

// After (fixed):
function Counter() {
  const [count, setCount] = useState(0);

  const increment = () => {
    setTimeout(() => {
      setCount(c => c + 1);  // Functional update uses latest state
    }, 1000);
  };

  return <button onClick={increment}>Count: {count}</button>;
}
```

**Pylint Coverage**: ❌ N/A (JavaScript)
**GNN Potential**: ✅ High (if trained on JS/TS)

---

### 8. **Flask Session Security Issues**
**Why Hard**: Requires understanding session management semantics
```python
# Before (session fixation vulnerability):
@app.route('/login', methods=['POST'])
def login():
    user = authenticate(request.form['username'], request.form['password'])
    if user:
        session['user_id'] = user.id  # Reuses existing session ID!
        return redirect('/dashboard')

# After (fixed):
@app.route('/login', methods=['POST'])
def login():
    user = authenticate(request.form['username'], request.form['password'])
    if user:
        session.clear()  # Clear old session
        session.regenerate()  # Generate new session ID
        session['user_id'] = user.id
        return redirect('/dashboard')
```

**Pylint Coverage**: ❌ None
**GNN Potential**: ✅ High

---

## 🎯 Tier 3: Algorithmic & Performance

### 9. **Accidental O(n²) in Pandas**
**Why Hard**: Requires understanding Pandas internals
```python
# Before (O(n²)):
df = pd.DataFrame()
for row in data:
    df = df.append(row, ignore_index=True)  # Creates new DataFrame each time

# After (O(n)):
df = pd.DataFrame(data)  # Single allocation
# OR
rows = []
for row in data:
    rows.append(row)
df = pd.DataFrame(rows)
```

**Pylint Coverage**: ❌ None
**GNN Potential**: ✅ Medium (pattern-based, not complexity analysis)

---

### 10. **Inefficient List Comprehension (Our Current Example)**
**Why Hard**: Requires complexity analysis
```python
# Before (O(n²)):
new_lines = [line for line in new_lines if line not in old_lines]

# After (O(n)):
old_set = set(old_lines)
new_lines = [line for line in new_lines if line not in old_set]
```

**Pylint Coverage**: ⚠️ Partial (has consider-using-set-comprehension but limited)
**GNN Potential**: ⚠️ Low (this is what pylint already does)

---

## 🎯 Tier 4: State Management & Architecture

### 11. **Singleton Implementation Bugs**
**Why Hard**: Requires understanding object lifecycle
```python
# Before (broken singleton):
class Config:
    _instance = None

    def __init__(self):
        Config._instance = self  # Bug: Allows multiple instances

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

# After (fixed):
class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

**Pylint Coverage**: ❌ None
**GNN Potential**: ✅ Medium

---

### 12. **Event Loop Blocking Operations**
**Why Hard**: Requires understanding async/await semantics
```python
# Before (blocks event loop):
async def download_file(url):
    response = requests.get(url)  # Blocking! Should use aiohttp
    return response.content

# After (fixed):
async def download_file(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()
```

**Pylint Coverage**: ❌ None (doesn't understand async semantics)
**GNN Potential**: ✅ High

---

## 📊 Category Priority for Curriculum Generation

### **Generate First** (Highest ROI):
1. ✅ Tainted Dataflow (SQL injection, XSS, command injection)
2. ✅ Race Conditions (shared state + threading)
3. ✅ Django N+1 Queries
4. ✅ Resource Leaks with Exceptions
5. ✅ Async/Await Misuse

### **Generate Second**:
6. ✅ Deadlock Patterns
7. ✅ Flask Security Issues
8. ✅ Pandas Performance Traps
9. ✅ Singleton Bugs
10. ✅ Event Loop Blocking

### **Skip Entirely** (Pylint Does This):
- ❌ Bare exception handling
- ❌ Unused imports
- ❌ Simple refactoring
- ❌ Basic error recovery
- ❌ Configuration management
- ❌ Most of the original 22 categories

---

## 🎯 Success Metrics

For each category, Nerion should:
1. **Find ≥3 real bugs** in Django/Flask that pylint missed
2. **Achieve ≥70% accuracy** on validation set
3. **Generalize** to similar patterns not in training (e.g., learn "N+1 in Django" → detect "N+1 in SQLAlchemy")

---

## 💀 What Went Wrong With Original Categories

| Original Category | Why It Failed | Pylint Coverage |
|-------------------|---------------|-----------------|
| "refactoring" | Too vague, mostly syntax | ✅ 90% |
| "bug_fixing" | Not specific enough | ✅ 80% |
| "error_recovery" | Just try/except patterns | ✅ 100% |
| "resource_management" | Only caught simple cases | ✅ 70% |
| "data_validation" | Type checkers do this | ✅ 90% |
| "configuration_management" | What does this even mean? | ❓ |

**New categories are 10x harder and pylint coverage is <20%.**

---

**THIS is what Nerion should have been trained on from day one.**
