# Checks qeval's graders without a model: correct reference solutions must pass every coding test, and plausible wrong
# ones must fail, so a score means something. Also prints the numeric answers and checks the instruction-following checks.
import json, sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qeval

GOOD = {
"c_merge": '''
def merge_intervals(intervals):
    out = []
    for s, e in sorted([list(x) for x in intervals]):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
''',
"c_expr": '''
def eval_expr(s):
    toks = []; i = 0
    while i < len(s):
        c = s[i]
        if c.isspace(): i += 1
        elif c in "+-*/()": toks.append(c); i += 1
        else:
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == "."): j += 1
            toks.append(float(s[i:j])); i = j
    pos = [0]
    def peek(): return toks[pos[0]] if pos[0] < len(toks) else None
    def nxt(): pos[0] += 1; return toks[pos[0] - 1]
    def expr():
        v = term()
        while peek() in ("+", "-"):
            v = v + term() if nxt() == "+" else v - term()
        return v
    def term():
        v = unary()
        while peek() in ("*", "/"):
            if nxt() == "*": v = v * unary()
            else: v = v / unary()
        return v
    def unary():
        if peek() == "-": nxt(); return -unary()
        if peek() == "(": nxt(); v = expr(); nxt(); return v
        return nxt()
    return expr()
''',
"c_osa": '''
def osa_distance(a, b):
    n, m = len(a), len(b)
    d = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): d[i][0] = i
    for j in range(m+1): d[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            c = 0 if a[i-1] == b[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+c)
            if i > 1 and j > 1 and a[i-1] == b[j-2] and a[i-2] == b[j-1]:
                d[i][j] = min(d[i][j], d[i-2][j-2]+1)
    return d[n][m]
''',
"c_topo": '''
import heapq
def smallest_topo_order(n, edges):
    adj = [[] for _ in range(n)]; indeg = [0]*n
    for u, v in edges: adj[u].append(v); indeg[v] += 1
    h = [i for i in range(n) if indeg[i] == 0]; heapq.heapify(h); out = []
    while h:
        u = heapq.heappop(h); out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0: heapq.heappush(h, v)
    if len(out) != n: raise ValueError("cycle")
    return out
''',
"c_lru": '''
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity): self.c = capacity; self.d = OrderedDict()
    def get(self, key):
        if key not in self.d: return -1
        self.d.move_to_end(key); return self.d[key]
    def put(self, key, value):
        if key in self.d: self.d.move_to_end(key)
        self.d[key] = value
        if len(self.d) > self.c: self.d.popitem(last=False)
    def keys(self): return list(reversed(self.d.keys()))
''',
"c_kth": '''
def kth_smallest_pair_distance(nums, k):
    a = sorted(nums); n = len(a); lo, hi = 0, a[-1] - a[0]
    while lo < hi:
        mid = (lo + hi) // 2; cnt = 0; j = 0
        for i in range(n):
            while a[i] - a[j] > mid: j += 1
            cnt += i - j
        if cnt >= k: hi = mid
        else: lo = mid + 1
    return lo
''',
"c_paren": '''
def longest_valid_parentheses(s):
    st = [-1]; best = 0
    for i, ch in enumerate(s):
        if ch == "(": st.append(i)
        else:
            st.pop()
            if not st: st.append(i)
            else: best = max(best, i - st[-1])
    return best
''',
"c_window": '''
from collections import Counter
def min_window(s, t):
    if not t: return ""
    need = Counter(t); missing = len(t); l = 0; best = (10**9, 0)
    for r, ch in enumerate(s):
        if need[ch] > 0: missing -= 1
        need[ch] -= 1
        if missing == 0:
            while need[s[l]] < 0: need[s[l]] += 1; l += 1
            if r - l + 1 < best[0]: best = (r - l + 1, l)
            need[s[l]] += 1; missing += 1; l += 1
    return "" if best[0] == 10**9 else s[best[1]:best[1] + best[0]]
''',
}

BAD = {
"c_merge": ("touching intervals not merged", GOOD["c_merge"].replace("s <= out[-1][1]", "s < out[-1][1]")),
"c_expr": ("uses eval with Python's rules: fails on 1/0 type", "def eval_expr(s):\n    return float(eval(s)) if '/0' not in s.replace(' ', '') else 1.0\n"),
"c_osa": ("plain Levenshtein, no transposition", GOOD["c_osa"].replace("if i > 1 and j > 1 and a[i-1] == b[j-2] and a[i-2] == b[j-1]:", "if False:")),
"c_topo": ("no cycle check", GOOD["c_topo"].replace('if len(out) != n: raise ValueError("cycle")', "pass")),
"c_lru": ("get does not count as a use", GOOD["c_lru"].replace("self.d.move_to_end(key); return self.d[key]", "return self.d[key]")),
"c_kth": ("O(n^2)", "def kth_smallest_pair_distance(nums, k):\n    return sorted(abs(x - y) for i, x in enumerate(nums) for y in nums[i+1:])[k-1]\n"),
"c_paren": ("counts all matched pairs instead of the longest run", "def longest_valid_parentheses(s):\n    o = c = 0\n    for ch in s:\n        if ch == '(': o += 1\n        elif o: o -= 1; c += 1\n    return 2 * c\n"),
"c_window": ("ignores multiplicity", "def min_window(s, t):\n    best = ''\n    for i in range(len(s)):\n        for j in range(i + 1, len(s) + 1):\n            if set(t) <= set(s[i:j]) and t and (best == '' or j - i < len(best)): best = s[i:j]\n    return best\n"),
}

tests = {n: t for n, p, t in qeval.CODE}
bad = 0
for name, code in GOOD.items():
    ok, detail = qeval.run_tests(code, tests[name])
    print("%-9s good: %s %s" % (name, "PASS" if ok else "FAIL", detail))
    bad += not ok
for name, (why, code) in BAD.items():
    ok, detail = qeval.run_tests(code, tests[name])
    print("%-9s wrong (%s): %s %s" % (name, why, "passes(!)" if ok else "fails, as it should", detail))
    bad += ok
print("numeric answers:", {n: a for n, p, a in qeval.NUMERIC})
IFCASES = [
 (qeval.check_four, "The sea is wide. Waves roll to the shore. Gulls cry above the foam. The ship sails toward the horizon.", True),
 (qeval.check_four, "The sea is wide, and deep. Waves roll. Gulls cry. Ships sail to the horizon.", False),
 (qeval.check_four, "Sure. Here are four sentences. The sea is wide. Waves roll in. Ships sail to the horizon.", False),
 (qeval.check_fruits, "- apple\n- banana\n- cherry\n- mango\n- kiwi", True),
 (qeval.check_fruits, "- Apple\n- banana\n- cherry\n- mango\n- kiwi", False),
 (qeval.check_fruits, "- apple\n- banana\n- cherry\n- mango", False),
 (qeval.check_json, '{"name": "Ana", "age": 31}', True),
 (qeval.check_json, '```json\n{"name": "Ana", "age": 31}\n```', False),
 (qeval.check_json, '{"name": "Ana", "age": 55}', False),
]
for fn, text, want in IFCASES:
    got = fn(text)
    print("%-13s want %-5s got %-5s %s" % (fn.__name__, want, got, "" if got == want else "<-- WRONG"))
    bad += got != want
print("last_int:", qeval.last_int("so ... ANSWER: 1,234"), qeval.last_int("the result is 42."), qeval.last_int("ANSWER: **-7**"))
print("extract_code:", repr(qeval.extract_code("text\n```python\nx = 1\n```\nmore\n```python\ny = 2\n```")))
print("PROBLEMS:", bad)
sys.exit(1 if bad else 0)
