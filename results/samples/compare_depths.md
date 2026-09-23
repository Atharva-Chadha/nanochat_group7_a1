# Model comparison: base:d6, base:d10

temperature 0.00, top-k 50, max 48 tokens, seed 42. Base models continue the prompt as raw text; SFT models receive it through the chat template.

| checkpoint | depth | d_model | heads | params | step | val bpb |
|---|---|---|---|---|---|---|
| `base:d6` | 6 | 384 | 3 | 73,531,538 | 1062 | 1.0129 |
| `base:d10` | 10 | 640 | 5 | 195,952,986 | 1605 | 0.8878 |

### The capital of France is

| checkpoint | continuation |
|---|---|
| `base:d6` | the capital of the world. The capital of France is the capital of the world. The capital of France is the capital of the world. The capital of France is the capital of the world. The capital of France is the capital of the |
| `base:d10` | the capital of the world. It is the capital of the world. It is the capital of the world. It is the capital of the world. It is the capital of the world. It is the capital of the world. It is |

### The chemical symbol of gold is

| checkpoint | continuation |
|---|---|
| `base:d6` | the gold symbol. It is a symbol of purity and purity. It is a symbol of purity and innocence. It is a symbol of purity and innocence. It is a symbol of purity and innocence. It is a symbol of |
| `base:d10` | Au. The symbol of gold is Au. The symbol of gold is Au. The symbol of gold is Au. The symbol of gold is Au. The symbol of gold is Au. The symbol of gold is Au. The symbol of gold |

### The planets of the solar system are

| checkpoint | continuation |
|---|---|
| `base:d6` | not only the largest and most abundant, but also the largest and most abundant. The planets of the solar system are also the largest and most abundant, and the largest and most abundant. The planets of the solar system are also the largest and |
| `base:d10` | the most massive bodies in the solar system. They are the most massive bodies in the solar system. They are the most massive bodies in the solar system. They are the most massive bodies in the solar system. They are the most massive bodies |

### The opposite of hot is

| checkpoint | continuation |
|---|---|
| `base:d6` | the hot. The hot is the hot. The hot is the hot. The hot is the hot. The hot is the hot. The hot is the hot. The hot is the hot. The hot is the hot. The hot is |
| `base:d10` | the cold. The cold is the opposite of the hot. The cold is the opposite of the hot. The cold is the opposite of the hot. The cold is the opposite of the hot. The cold is the opposite of the hot. |

### If yesterday was Friday, then tomorrow will be

| checkpoint | continuation |
|---|---|
| `base:d6` | Friday. The day was 10:00 a.m. and 10:00 a.m. The day was 10:00 a.m. and 10:00 a.m. The day was 10:00 a.m. |
| `base:d10` | Saturday. If tomorrow was Sunday, then tomorrow will be Sunday. If tomorrow was Sunday, then tomorrow will be Sunday. If tomorrow was Sunday, then tomorrow will be Sunday. If tomorrow was Sunday, then tomorrow will be Sunday. If tomorrow |

### If 5*x + 3 = 13, then x is

| checkpoint | continuation |
|---|---|
| `base:d6` | the number of times the number of times the number of times the number of times the number of times the number of times the number of times the number of times the number of times the number of times the number of times the number of times |
| `base:d10` | the number of moles of the element. If 5*x + 3 = 13, then x is the number of moles of the element. If 5*x + 3 = 13, then x is the number |

### Water boils at a temperature of

| checkpoint | continuation |
|---|---|
| `base:d6` | 100 degrees Celsius (100 degrees Fahrenheit) and 100 degrees Fahrenheit (100 degrees Celsius) at 100 degrees Celsius (100 degrees Fahrenheit). The temperature of the water is 100 degrees Celsius (10 |
| `base:d10` | 1,000 degrees Fahrenheit. The temperature of water is 1,000 degrees Fahrenheit. The temperature of water is 1,000 degrees Fahrenheit. The temperature of water is 1,000 degrees Fahrenheit. The |

### The largest ocean on Earth is

| checkpoint | continuation |
|---|---|
| `base:d6` | the largest ocean on Earth. The largest ocean on Earth is the largest ocean on Earth. The largest ocean on Earth is the largest ocean on Earth. The largest ocean on Earth is the largest ocean on Earth. The largest ocean on Earth is |
| `base:d10` | the Mariana Trench, which is 1,000 miles long and 1,000 miles wide. It is the largest ocean in the world, and it is home to a variety of marine life, including whales, dolphins |

