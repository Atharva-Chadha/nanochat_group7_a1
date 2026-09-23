# Tokenizer analysis (Task 1)

Tokenizers: ours-8k / ours-32k = nanochat rustbpe BPE trained on 500M chars of ClimbMix (`10_train_tokenizers.sh`), GPT-2 (50,257) and GPT-4 cl100k_base (100,277) via tiktoken for reference.

## Compression and sequence length

| text | bytes | ours-8k tokens | ours-32k tokens | gpt2-50k tokens | gpt4-100k tokens | ours-8k bytes/tok | ours-32k bytes/tok | gpt2-50k bytes/tok | gpt4-100k bytes/tok | 8k/32k length ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| English (own) | 396 | 92 | 74 | 74 | 74 | 4.30 | 5.35 | 5.35 | 5.35 | 1.24x |
| Python code | 377 | 175 | 151 | 162 | 123 | 2.15 | 2.50 | 2.33 | 3.07 | 1.16x |
| Numbers | 183 | 105 | 100 | 82 | 91 | 1.74 | 1.83 | 2.23 | 2.01 | 1.05x |
| Korean | 134 | 134 | 122 | 118 | 51 | 1.00 | 1.10 | 1.14 | 2.63 | 1.10x |
| German | 112 | 58 | 47 | 44 | 34 | 1.93 | 2.38 | 2.55 | 3.29 | 1.23x |
| Hindi | 147 | 147 | 91 | 87 | 58 | 1.00 | 1.62 | 1.69 | 2.53 | 1.62x |
| Emoji/accents | 75 | 41 | 31 | 24 | 24 | 1.83 | 2.42 | 3.12 | 3.12 | 1.32x |
| ClimbMix val (2M chars) | 2,002,641 | 500,045 | 426,451 | 430,868 | 416,971 | 4.00 | 4.70 | 4.65 | 4.80 | 1.17x |

tokens per character (the spec's 'compression ratio'): English (own): 8k 0.232, 32k 0.187; Python code: 8k 0.464, 32k 0.401; Numbers: 8k 0.574, 32k 0.546; Korean: 8k 2.481, 32k 2.259; German: 8k 0.542, 32k 0.439; Hindi: 8k 2.579, 32k 1.596; Emoji/accents: 8k 0.695, 32k 0.525; ClimbMix val (2M chars): 8k 0.250, 32k 0.213

## Vocabulary composition

| tokenizer | vocab | mean token bytes | 1-byte tokens | tokens >= 8 bytes | digit-only tokens | tokens with non-ASCII bytes |
|---|---|---|---|---|---|---|
| ours-8k | 8,192 | 5.54 | 256 | 1,800 | 108 | 165 |
| ours-32k | 32,768 | 6.56 | 256 | 11,290 | 112 | 410 |

longest tokens (ours-8k): `----------------`, ` recommendations`, ` characteristics`, ` transportation`, ` sustainability`, ` responsibility`, ` infrastructure`, ` implementation`, ` identification`, ` characteristic`, ` understanding`, ` technological`

longest tokens (ours-32k): `________________________________`, `................................`, `--------------------------------`, `……………………`, ` telecommunications`, ` neurotransmitters`, ` multidisciplinary`, ` misunderstandings`, ` interdisciplinary`, ` responsibilities`, ` neurotransmitter`, ` misunderstanding`

## Probes: how numbers, code, whitespace, non-English text and emoji are split

| probe | ours-8k | ours-32k | gpt4-100k |
|---|---|---|---|
| `1234567890` | 5: `12` `34` `56` `78` `90` | 5: `12` `34` `56` `78` `90` | 4: `123` `456` `789` `0` |
| `3.14159265` | 6: `3` `.` `14` `15` `92` `65` | 6: `3` `.` `14` `15` `92` `65` | 5: `3` `.` `141` `592` `65` |
| `2026-09-21` | 6: `20` `26` `-` `09` `-` `21` | 6: `20` `26` `-` `09` `-` `21` | 6: `202` `6` `-` `09` `-` `21` |
| `$1,234.56` | 7: `$` `1` `,` `23` `4` `.` `56` | 7: `$` `1` `,` `23` `4` `.` `56` | 6: `$` `1` `,` `234` `.` `56` |
| `0x7FFF` | 6: `0` `x` `7` `F` `F` `F` | 5: `0` `x` `7` `FF` `F` | 4: `0` `x` `7` `FFF` |
| `def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)` | 27: `d` `ef` ` fib` `(` `n` `):` ` return` ` n` ` if` ` n` ` <` ` ` `2` ` else` ` fib` `(` `n` `-` `1` `)` ` +` ` fib` `(` `n` `-` `2` `)` | 23: `def` ` fib` `(n` `):` ` return` ` n` ` if` ` n` ` <` ` ` `2` ` else` ` fib` `(n` `-` `1` `)` ` +` ` fib` `(n` `-` `2` `)` | 23: `def` ` fib` `(n` `):` ` return` ` n` ` if` ` n` ` <` ` ` `2` ` else` ` fib` `(n` `-` `1` `)` ` +` ` fib` `(n` `-` `2` `)` |
| `    x = 1\n        y = 2\n` | 13: `   ` ` x` ` =` ` ` `1` `\n` `    ` `   ` ` y` ` =` ` ` `2` `\n` | 12: `   ` ` x` ` =` ` ` `1` `\n` `       ` ` y` ` =` ` ` `2` `\n` | 12: `   ` ` x` ` =` ` ` `1` `\n` `       ` ` y` ` =` ` ` `2` `\n` |
| `안녕하세요 세계` | 22: `�` `�` `�` `�` `�` `�` `�` `�` `�` `�` `�` `�` `�` `�` `�` ` ` `�` `�` `�` `�` `�` `�` | 20: `�` `�` `�` `�` `�` `�` `�` `�` `�` `�` `�` `�` `�` `�` ` �` `�` `�` `�` `�` `�` | 9: `�` `�` `�` `�` `하세요` ` �` `�` `�` `�` |
| `Straße` | 5: `St` `ra` `�` `�` `e` | 5: `St` `ra` `�` `�` `e` | 2: `Stra` `ße` |
| `naïve café` | 7: `na` `�` `�` `ve` ` c` `af` `é` | 5: `na` `ï` `ve` ` caf` `é` | 4: `na` `ï` `ve` ` café` |
| `🚀🌍😀` | 9: `�` `�` `�` `�` `�` `�` `�` `�` `�` | 9: `�` `�` `�` `�` `�` `�` `�` `�` `�` | 8: `�` `�` `�` `�` `�` `�` `�` `�` |
| `supercalifragilisticexpialidocious` | 15: `s` `u` `per` `cal` `if` `rag` `il` `ist` `ice` `x` `p` `ial` `id` `oc` `ious` | 12: `super` `cal` `if` `rag` `il` `ist` `ice` `xp` `ial` `id` `oc` `ious` | 11: `sup` `erc` `al` `if` `rag` `il` `istic` `exp` `ial` `id` `ocious` |
| `unbelievably` | 6: `un` `b` `el` `ie` `v` `ably` | 4: `un` `bel` `iev` `ably` | 3: `un` `belie` `vably` |

![compression](../report/figures/tokenizer_compression.png)

