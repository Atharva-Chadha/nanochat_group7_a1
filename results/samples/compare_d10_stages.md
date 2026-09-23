# Model comparison: base:d10, sft:d10

temperature 0.70, top-k 50, max 48 tokens, seed 42. Base models continue the prompt as raw text; SFT models receive it through the chat template.

| checkpoint | depth | d_model | heads | params | step | val bpb |
|---|---|---|---|---|---|---|
| `base:d10` | 10 | 640 | 5 | 195,952,986 | 1605 | 0.8878 |
| `sft:d10` | 10 | 640 | 5 | 195,952,986 | 934 | 0.3851 |

### What is the capital of France?

| checkpoint | continuation |
|---|---|
| `base:d10` | The capital of France, Lansdow, is the capital of the Republic of France. The capital is located at the mouth of the Nile, along the Nile River, and is surrounded by a sea of volcanic rocks. Lansd |
| `sft:d10` | The capital of France, Paris, is Paris. It is a city located in the northern part of France and is famous for its iconic landmarks, such as the Eiffel Tower, the Louvre Museum, and Notre-Dame |

### Write a short poem about the sea.

| checkpoint | continuation |
|---|---|
| `base:d10` | If you write a short poem in a line, you can include several adjectives, and you can use a variety of them to create a short poem. How to write a poem in a line? To Write a Short Poem On How |
| `sft:d10` | In the depths of darkness, where waves crash, A lighthouse stands, a beacon's light, The sound of its gentle lapping, A symphony of sound, a symphony of rhyme. In the ocean's depths, where seas drift |

### Why is the sky blue?

| checkpoint | continuation |
|---|---|
| `base:d10` | The sky is blue, the sea is blue, and the sun is blue. The sky is blue. The sky is blue. The sky is blue. The sky is blue. The sky is blue. The sky is blue. The sky |
| `sft:d10` | The sky blue is primarily due to the uneven distribution of light around the Earth by the Earth's atmosphere and oceans. This phenomenon is primarily caused by the tilt of the Earth's axis, which is approximately 23.5 degrees. The amount |

### If I have 3 apples and buy 4 more, how many apples do I have?

| checkpoint | continuation |
|---|---|
| `base:d10` | I have 3 more apples so I have 4 more apples! I know I need 5 more apples, but I have 3 more apples too, so I have 4 more apples so I need 5 more apples so I |
| `sft:d10` | To find out how many apples you have, let's break down the information given. We have 3 apples and 4 more, so we need to subtract the number of apples you purchased from the number of apples you have. This |

### Explain in two sentences what a transformer neural network is.

| checkpoint | continuation |
|---|---|
| `base:d10` | Explain in two sentences in two sentences what a transformer neural network is. Explain in two sentences what a transformer neural network is. Explain in two sentences in two sentences what a transformer neural network is. What is an AC transformer? How does a |
| `sft:d10` | A transformer neural network (NNP) is a type of artificial neural network that uses a self-attention mechanism to capture and process input data. Unlike traditional neural networks, which rely on labeled data, a transformer has a self-attention mechanism |

