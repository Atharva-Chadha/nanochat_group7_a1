# Model comparison: sft:d10

temperature 0.70, top-k 50, max 128 tokens, seed 42. Base models continue the prompt as raw text; SFT models receive it through the chat template.

| checkpoint | depth | d_model | heads | params | step | val bpb |
|---|---|---|---|---|---|---|
| `sft:d10` | 10 | 640 | 5 | 195,952,986 | 934 | 0.3851 |

### What is the capital of France?

| checkpoint | continuation |
|---|---|
| `sft:d10` | The capital of France, Paris, is Paris. It is a city located in the northern part of France and is famous for its iconic landmarks, such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. Paris is also known for its charming neighborhoods, like Montmartre and Le Marais, which are filled with artists, street performers, and cafes, making it a vibrant and eclectic city. |

### Write a short poem about the sea.

| checkpoint | continuation |
|---|---|
| `sft:d10` | In the depths of darkness, where waves crash, A lighthouse stands, a beacon's light, The sound of its gentle lapping, A symphony of sound, a symphony of rhyme. In the ocean's depths, where seas drift, A sea of waves, a sea of ripples, A rhythm that pulses, a heartbeat of calm, A heartbeat that speaks, a heartbeat that sways. The wind stirs, a gentle breeze, A slow, steady heartbeat, a soothing sight, A heartbeat of sound, a heartbeat that echoes, A heartbeat that echoes, a heartbeat that sways. In the heart of the |

### Why is the sky blue?

| checkpoint | continuation |
|---|---|
| `sft:d10` | The sky blue is primarily due to the uneven distribution of light around the Earth by the Earth's atmosphere and oceans. This phenomenon is primarily caused by the tilt of the Earth's axis, which is approximately 23.5 degrees. The amount of solar radiation received by the Earth's surface is approximately 18.5 degrees per year, which is roughly the amount of light that is absorbed or scattered by the atmosphere during the Northern Hemisphere's summer and winter. This uneven distribution of light allows the Earth's atmosphere to play a significant role in shaping the color we see. The Earth's orbit around the Sun is tilted at an angle of about |

### If I have 3 apples and buy 4 more, how many apples do I have?

| checkpoint | continuation |
|---|---|
| `sft:d10` | To find out how many apples you have, let's break down the information given. We have 3 apples and 4 more, so we need to subtract the number of apples you purchased from the number of apples you have. This gives us 3 - 4 = 2 apples. We then subtract this from the number of apples you have, which is 2. So, you have 2 apples. Now, we need to find out how many apples you have, so we subtract this from the number of apples you have. 2 apples - 2 apples = 1 apple. Therefore, you have |

### Explain in two sentences what a transformer neural network is.

| checkpoint | continuation |
|---|---|
| `sft:d10` | A transformer neural network (NNP) is a type of artificial neural network that uses a self-attention mechanism to capture and process input data. Unlike traditional neural networks, which rely on labeled data, a transformer has a self-attention mechanism that allows the network to capture and store input data in sequential units, such as images or text. This makes it ideal for tasks like image recognition, text processing, and text classification, where the goal is to extract relevant information from the input data. |

