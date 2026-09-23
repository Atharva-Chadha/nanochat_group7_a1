# Model comparison: sft:d10

temperature 0.10, top-k 50, max 128 tokens, seed 42. Base models continue the prompt as raw text; SFT models receive it through the chat template.

| checkpoint | depth | d_model | heads | params | step | val bpb |
|---|---|---|---|---|---|---|
| `sft:d10` | 10 | 640 | 5 | 195,952,986 | 934 | 0.3851 |

### What is the capital of France?

| checkpoint | continuation |
|---|---|
| `sft:d10` | The capital of France is Paris. Paris is a city located in the northern part of France, near the Seine River and the French Riviera. It is a major economic and cultural center, known for its rich history, art, architecture, and cuisine. Paris is also home to the famous Eiffel Tower, the Louvre Museum, and the Palace of Versailles. |

### Write a short poem about the sea.

| checkpoint | continuation |
|---|---|
| `sft:d10` | In the depths of the ocean, where the waves crash, A symphony of sound, a symphony of light, A symphony of the sea, a symphony of sound. The waves crash against the shore, a crashing wave, A soothing melody, a symphony of sound, A symphony of the sea, a symphony of sound. The tides, a gentle breeze, a soothing melody, A symphony of the sea, a symphony of sound, A symphony of the sea, a symphony of sound. The waves crash against the shore, a crashing wave, A soothing melody, a symphony of sound, A symphony of the sea, a symphony |

### Why is the sky blue?

| checkpoint | continuation |
|---|---|
| `sft:d10` | The sky blue is a phenomenon that occurs because the Earth's atmosphere contains a layer of gases that are not visible to the naked eye. These gases are primarily composed of nitrogen and oxygen, which are the building blocks of life. When sunlight enters the Earth's atmosphere, it encounters these gases, causing them to scatter and change color. This is because the Earth's atmosphere is composed of nitrogen and oxygen, and the gases in the atmosphere are not visible to the naked eye. As a result, the sky blue is often seen in the sky during the day, when the sun is low in the sky and the Earth's atmosphere is not visible to |

### If I have 3 apples and buy 4 more, how many apples do I have?

| checkpoint | continuation |
|---|---|
| `sft:d10` | To find out how many apples you have, let's break down the information you have. You have 3 apples and buy 4 more, which means you have 3 + 4 = 7 apples. Since you have 7 apples, you can calculate the number of apples you have by dividing the total number of apples by the number of apples you have. 7 apples / 7 apples per apple = 3 apples per apple. So, you have 3 apples and 3 apples. |

### Explain in two sentences what a transformer neural network is.

| checkpoint | continuation |
|---|---|
| `sft:d10` | A transformer neural network is a type of neural network that uses self-attention mechanisms to weigh the importance of different weights in a model. It's a type of neural network that consists of two main components: the encoder and the decoder. The encoder takes in a set of input features and produces a output vector, which is a weighted average of the input features. The decoder then uses this weighted average to make predictions about the input features, taking into account the weights of the input features. This process is repeated for each input feature, allowing the network to learn the relationships between the input features and the output features. The transformer's output |

