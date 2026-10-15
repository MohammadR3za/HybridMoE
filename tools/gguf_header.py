#!/usr/bin/env python3
"""Print a GGUF file's metadata and tensor list from its header alone.

gguf-py maps every tensor's data, so it refuses a file that is still being
downloaded. The header (metadata, then one record per tensor) sits at the
start of the file and is complete long before the data is, so this reads only
that part.

usage: gguf_header.py FILE [--tensors PATTERN]
"""
import re
import struct
import sys

TYPES = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
GGML_TYPES = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1',
              10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS',
              17: 'IQ2_XS', 18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_S',
              23: 'IQ4_XS', 24: 'I8', 25: 'I16', 26: 'I32', 27: 'I64', 28: 'F64', 29: 'IQ1_M', 30: 'BF16',
              39: 'MXFP4'}


def main():
    path = sys.argv[1]
    pattern = re.compile(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == '--tensors' else None
    with open(path, 'rb') as f:
        def rd(fmt):
            return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]

        def rstr():
            return f.read(rd('Q')).decode('utf-8', 'replace')

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                et, n = rd('I'), rd('Q')
                if et == 8:
                    return [rstr() for _ in range(n)]
                return [rd(TYPES[et]) for _ in range(n)]
            return rd(TYPES[t])

        assert f.read(4) == b'GGUF', 'not a GGUF file'
        version, n_tensors, n_kv = rd('I'), rd('Q'), rd('Q')
        print(f'version {version}, {n_tensors} tensors, {n_kv} keys')
        for _ in range(n_kv):
            key = rstr()
            val = rval(rd('I'))
            if key.startswith('tokenizer.ggml.') and key not in ('tokenizer.ggml.model', 'tokenizer.ggml.pre'):
                continue
            if isinstance(val, list) and len(val) > 8:
                val = f'{val[:8]}... ({len(val)} items)'
            print(f'  {key} = {val}')
        for _ in range(n_tensors):
            name = rstr()
            dims = [rd('Q') for _ in range(rd('I'))]
            gtype, _offset = rd('I'), rd('Q')
            if pattern is None or pattern.search(name):
                print(f'  {name:40s} {GGML_TYPES.get(gtype, gtype):8s} {dims}')


if __name__ == '__main__':
    main()
