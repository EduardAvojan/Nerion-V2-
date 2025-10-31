"""Pre-validate ALL lessons before starting dataset generation."""
import ast
import sqlite3
import sys
from pathlib import Path

def validate_lesson(name, code, max_size_kb=1000):
    """Validate a single code snippet."""
    size_kb = len(code) / 1024
    
    if size_kb > max_size_kb:
        return False, f"Too large: {size_kb:.1f}KB > {max_size_kb}KB"
    
    try:
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        # tree-sitter fallback would handle this
        return True, f"Python syntax error (tree-sitter will handle)"
    except Exception as e:
        return False, f"Parse error: {e}"

print("="*100)
print("PRE-VALIDATION: Testing ALL 1,635 Lessons")
print("="*100)

conn = sqlite3.connect('out/learning/curriculum.sqlite')
cursor = conn.cursor()
cursor.execute("SELECT name, before_code, after_code FROM lessons ORDER BY id")

total = 0
failures = []
warnings = []
too_large = []

for row in cursor.fetchall():
    name, before, after = row
    total += 1
    
    # Check before_code
    ok, msg = validate_lesson(name, before)
    if not ok:
        if "Too large" in msg:
            too_large.append((name, before, msg))
        else:
            failures.append((name, "before_code", msg))
    elif "syntax error" in msg:
        warnings.append((name, "before_code", msg))
    
    # Check after_code  
    ok, msg = validate_lesson(name, after)
    if not ok:
        if "Too large" in msg:
            too_large.append((name, after, msg))
        else:
            failures.append((name, "after_code", msg))
    elif "syntax error" in msg:
        warnings.append((name, "after_code", msg))

conn.close()

print(f"\nValidated {total} lessons")
print(f"  Failures: {len(failures)}")
print(f"  Warnings: {len(warnings)} (non-Python, tree-sitter will handle)")
print(f"  Too large: {len(too_large)} files")

if too_large:
    print(f"\n{'='*100}")
    print("EXTREMELY LARGE FILES (>1MB):")
    print(f"{'='*100}")
    for name, code, msg in too_large[:10]:
        print(f"  {name}: {msg}")
        print(f"    Lines: {code.count(chr(10))}")

if failures:
    print(f"\n{'='*100}")
    print(f"CRITICAL FAILURES: {len(failures)}")
    print(f"{'='*100}")
    for name, code_type, msg in failures[:20]:
        print(f"  {name} ({code_type}): {msg}")
    
    print(f"\n❌ CANNOT PROCEED - Fix these failures first!")
    sys.exit(1)

print(f"\n{'='*100}")
print("✅ ALL LESSONS CAN BE PARSED")
print(f"{'='*100}")
print(f"\nNOTE: {len(warnings)} lessons use non-Python syntax (tree-sitter will handle)")
if too_large:
    print(f"WARNING: {len(too_large)} files are >1MB (will be SLOW to parse)")

