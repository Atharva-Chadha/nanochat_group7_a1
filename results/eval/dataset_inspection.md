# Dataset inspection (Task 3 sanity check)

## Number of training examples

| dataset | rows |
|---|---|
| SmolTalk train | 460,341 |
| MMLU auxiliary_train | 99,842 |
| GSM8K train | 7,473 |
| **Stage 1 mixture (MMLU x3 + GSM8K x4)** | 329,418 |
| **Stage 2 mixture (SmolTalk x1)** | 460,341 |
| upstream nanochat mixture (all three) | 789,759 |

## First rows (as the conversation dicts the trainer sees)

### SmolTalk train

**example 0** ```json
[
 {
  "content": "A researcher interested in examining students' study time per day collected data from a random sample of 20 students. Here are the data: \n\n3, 1, 2, 4, 5, 2, 3, 4, 2, 1, 2, 3, 4, 6, 3, 2, 2, 1, 5, 4\n\nWhat is the mean of this dataset? How would you describe or interpret the result?",
  "role": "user"
 },
 {
  "content": "To calculate the mean, we need to add up all the values and divide by the total number of observations. \n\nThe sum of the values is 3 + 1 + 2 + 4 + 5 + 2 + 3 + 4 + 2 + 1 + 2 + 3 + 4 + 6 + 3 + 2 + 2 + 1 + 5 + 4 = 59.\n\nThere are 20 observations in the dataset. The mean is calculated by dividing the sum by the number of observations: 59 / 20 = 2.95.\n\nThe mean of 2.95 hours per day indicates that, on average, students in this sample spend approximately 3 hours per day studying. This result suggests a relatively moderate amount of daily study time, with most students likely dedicating a few hours to studying each day. However, to better understand the study habits of this sample, further analysis, such as examining the distribution of study times or identifying any outliers, may be necessary.",
  "role": "assistant"
 },
 {
  "content": "How would you describe the dispersion of this data set. Which measurement or calculation of dispersion would you use to describe and or interpret the dispersion of the data.",
  "role": "user"
 },
 {
  "content": "To describe the dispersion of the data, I would calculate the standard deviation, as it is a 
```
**example 1** ```json
[
 {
  "content": "You're an AI assistant for text re-writing. Rewrite the input text to make it more professional and formal while retaining its essential content.",
  "role": "system"
 },
 {
  "content": "Hey Alex,\n\nSo, I've been thinking a lot about our new project on the etymology of ancient languages, and I have a ton of ideas! I was reading this fascinating article about the Proto-Indo-European roots of some modern words, and it got me thinking about how we could really dive deep into this. Like, we could look at how certain words have evolved over time and see if there are any patterns or commonalities. \n\nI know it's a lot to take in, but I think it could be really interesting. Let me know what you think, and maybe we can set up a time to discuss it further. Oh, and don't forget about the conference next month! 😊\n\nTalk soon!\nEleanor",
  "role": "user"
 },
 {
  "content": "Dear Alex,\n\nI hope this message finds you well. I have been giving considerable thought to our upcoming project on the etymology of ancient languages and have developed several ideas that I believe could enhance our research. Recently, I came across an insightful article discussing the Proto-Indo-European roots of various modern words. This has inspired me to consider a more in-depth exploration of how certain words have evolved over time, potentially uncovering patterns or commonalities that could be significant to our study.\n\nI understand that this is a substantial area of inquiry, but I 
```
**example 2** ```json
[
 {
  "content": "Implement a Python function that takes a sequence of integers and outputs the longest non-decreasing contiguous subsequence of the input.\n\n```python\ndef longest_non_decreasing_subsequence(sequence: List[int]) -> List[int]:\n    \"\"\"\n    Find the longest non-decreasing subsequence of the given sequence.\n\n    Args:\n        sequence (List[int]): The input sequence of integers.\n\n    Returns:\n        List[int]: The longest non-decreasing subsequence of the input sequence.\n    \"\"\"\n    # Your implementation goes here\n    ...\n```\n\nThe function should handle empty sequences, negative numbers, and decreasing sequences correctly. The function should return an empty list if the input sequence is empty.\n\nYour code should pass the following assertions:\n```python\nassert longest_non_decreasing_subsequence([1, 2, 3, 5, 2, 4]) == [1, 2, 3, 5]\nassert longest_non_decreasing_subsequence([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]\nassert longest_non_decreasing_subsequence([]) == []\nassert longest_non_decreasing_subsequence([-1, -2, -3]) == [-1]\n```",
  "role": "user"
 },
 {
  "content": "To find the longest non-decreasing contiguous subsequence, we can use a dynamic programming approach. We iterate through the input sequence, and for each element, we compare it with the previous element. If the current element is greater than or equal to the previous element, we add it to the current subsequence. If it is smaller, we start a new subsequence with the current 
```
### MMLU auxiliary_train

**example 0** ```json
[
 {
  "role": "user",
  "content": "Multiple Choice question: Rules in the reading room Hello, everyone. Welcome to the school reading room. We hope you have a good time here. Before you go into the reading room, there are some rules you need to keep. 1.The reading room is open from 8:00 a.m. to 5:00 p.m. from Monday to Friday. 2. Don't take your bag into the reading room. 3. Don't talk loudly in the reading room. 4. Don't take any food or drink into the reading room. 5. Take only one book at a time. After you finish reading the book, you must put it back and then you can take another one. Don't take many books to your seat. 6. Before you leave, you must the book to the bookshelf. You can't take any book out of the reading room. How long is the reading room open every day?\n- Ten hours.=A\n- Nine hours.=B\n- Seven hours.=C\n- Eight hours.=D\n\nRespond only with the letter of the correct answer."
 },
 {
  "role": "assistant",
  "content": "B"
 }
]
```
**example 1** ```json
[
 {
  "role": "user",
  "content": "Multiple Choice question: AFRICAN MANGO FAQs Q : How do I use AFRICAN MANGO? A : It's simple... Just take one rapid - release capsule  , 3 times daily before meals (i.e., 30 minutes before breakfast, lunch, and dinner.) Q : Is it safe? A : Yes. Unlike other diet pills, Pure African Mango Extract (lrvingia gabonensis) does not contain any natural or artificial stimulants. Clinical studies have shown this nutrient to be both safe and well-tolerated. It's easy on your system and helps you lose body fat naturally. Q : Who uses AFRICAN MANGO? A : Everybody from A-List Hollywood movie stars to everyday men and women who want to lose pounds and inches of excess body fat. It's currently popular in Hollywood today because many of today's movie scripts call for actors and actresses to be in their absolute best shape. Q : Do I have to exercise and diet in order to see results? A : Research shows 100% Pure African Mango Extract helped people lose weight and belly fat without additional diet or exercise. However, in order to achieve your best weight-loss results, we highly recommend incorporating fun lifestyle exercises (like walking, tennis, etc.) and replacing high-calorie foods with healthy nutrient-dense foods like fruits, vegetables, and whole grains, and lean proteins. Also, drinking more water daily speeds weight loss, according to a new study from Viginia Tech University. Q : When I place my order, it's a one-time-only transaction  , right? A :
```
**example 2** ```json
[
 {
  "role": "user",
  "content": "Multiple Choice question: Many years ago, children who had good manners kept quiet if their parents were talking with other persons. Today, well-mannered children have more freedom.  Sometimes good manners in one place are bad manners in other place.  Suppose you are a visitor in the prefix = st1 /landofMongolia. Some friends ask you to eat with them. What kind of manners do they want you to have? They want you to give a loud \"burp\" after you finish eating. Burping will show that you like your food. In some countries, if you give a loud burp, you are told to say, \"Excuse me, please.\" In many places people like to eat together. But in some parts of Polynesia, it is bad manners to be seen eating at all. People show their good manners by turning their back on others while they eat.  What are good manners like in an East African town? The people try not to see you. They are being polite. You may see a friend. He may not see you at all. If you're polite, you will sit down beside him. You will wait until he finishes what he is doing. Then he will talk to you.  Suppose you visit a friend in Arabia. You should walk behind the tents until you come to his tent. If you pass in front of the other tents, you will be asked into them. The people will ask you to eat with them. And it is bad manners if you say no. Which of the following sentences is not true according to the passage?\n- Well-mannered children should always keep quiet.=A\n- Eating with 
```
### GSM8K train

**example 0** ```json
[
 {
  "role": "user",
  "content": "Mimi picked up 2 dozen seashells on the beach.  Kyle found twice as many shells as Mimi and put them in his pocket. Leigh grabbed one-third of the shells that Kyle found.  How many seashells did Leigh have?"
 },
 {
  "role": "assistant",
  "content": [
   {
    "type": "text",
    "text": "Mimi has 2 x 12 = "
   },
   {
    "type": "python",
    "text": "2*12"
   },
   {
    "type": "python_output",
    "text": "24"
   },
   {
    "type": "text",
    "text": "24 sea shells.\nKyle has 24 x 2 = "
   },
   {
    "type": "python",
    "text": "24*2"
   },
   {
    "type": "python_output",
    "text": "48"
   },
   {
    "type": "text",
    "text": "48 sea shells.\nLeigh has 48 / 3 = "
   },
   {
    "type": "python",
    "text": "48/3"
   },
   {
    "type": "python_output",
    "text": "16"
   },
   {
    "type": "text",
    "text": "16 sea shells.\n#### 16"
   }
  ]
 }
]
```
**example 1** ```json
[
 {
  "role": "user",
  "content": "Frankie's parents let him have many pets. He has six more snakes than he has cats. He has one less parrot than cats. Six of his pets have four legs. He has 2 dogs. How many pets does he have in total?"
 },
 {
  "role": "assistant",
  "content": [
   {
    "type": "text",
    "text": "He has 6 - 2 = "
   },
   {
    "type": "python",
    "text": "6-2"
   },
   {
    "type": "python_output",
    "text": "4"
   },
   {
    "type": "text",
    "text": "4 cats.\nHe has 4 - 1 = "
   },
   {
    "type": "python",
    "text": "4-1"
   },
   {
    "type": "python_output",
    "text": "3"
   },
   {
    "type": "text",
    "text": "3 parrots.\nHe has 4 + 6 = "
   },
   {
    "type": "python",
    "text": "4+6"
   },
   {
    "type": "python_output",
    "text": "10"
   },
   {
    "type": "text",
    "text": "10 snakes.\nHe has a total of 2 + 4 + 3 + 10 = "
   },
   {
    "type": "python",
    "text": "2+4+3+10"
   },
   {
    "type": "python_output",
    "text": "19"
   },
   {
    "type": "text",
    "text": "19 pets.\n#### 19"
   }
  ]
 }
]
```
**example 2** ```json
[
 {
  "role": "user",
  "content": "Olaf collects colorful toy cars. At first, his collection consisted of 150 cars. His family, knowing his hobby, decided to give him some toy cars. Grandpa gave Olaf twice as many toy cars as the uncle. Dad gave Olaf 10 toy cars, 5 less than Mum. Auntie gave Olaf 6 toy cars, 1 more than the uncle. How many toy cars does Olaf have in total, after receiving all these gifts?"
 },
 {
  "role": "assistant",
  "content": [
   {
    "type": "text",
    "text": "Dad gave Olaf 10 toy cars,\nMom has given Olaf 5 more toy cars than Dad, so 10 + 5 = "
   },
   {
    "type": "python",
    "text": "10+5"
   },
   {
    "type": "python_output",
    "text": "15"
   },
   {
    "type": "text",
    "text": "15 toy cars\nAuntie gave Olaf 6 toy cars,\nUncle has given 1 less toy than Auntie, so 6 - 1 = "
   },
   {
    "type": "python",
    "text": "6-1"
   },
   {
    "type": "python_output",
    "text": "5"
   },
   {
    "type": "text",
    "text": "5 toy cars\nGrandpa gave Olaf 2 * 5 = "
   },
   {
    "type": "python",
    "text": "2*5"
   },
   {
    "type": "python_output",
    "text": "10"
   },
   {
    "type": "text",
    "text": "10 toy cars.\nAll the family together gave Olaf 10 +15 + 6 + 5 + 10 = "
   },
   {
    "type": "python",
    "text": "10+15+6+5+10"
   },
   {
    "type": "python_output",
    "text": "46"
   },
   {
    "type": "text",
    "text": "46.\nAdding the cars Olaf already had, Olaf's collection has 150 + 46 = "
   },
   {
    "typ
```

## Loss mask on one rendered example per dataset

`render_conversation` returns token ids and a mask; the SFT loss is computed only where mask = 1 (assistant text and `<|assistant_end|>`). BOS, user turns, `<|user_start|>`/`<|user_end|>`, `<|assistant_start|>` and python tool *outputs* are mask = 0 (targets set to -1 = ignore_index).

### SmolTalk train - 357 tokens, 170 supervised (47.6%)

| pos | token | mask |
|---|---|---|
| 0 | `<\|bos\|>` | 0 |
| 1 | `<\|user_start\|>` | 0 |
| 2 | `You` | 0 |
| 3 | `'re` | 0 |
| 4 | ` an` | 0 |
| 5 | ` AI` | 0 |
| 6 | ` assistant` | 0 |
| 7 | ` for` | 0 |
| 8 | ` text` | 0 |
| 9 | ` re` | 0 |
| 10 | `-w` | 0 |
| 11 | `riting` | 0 |
| 12 | `.` | 0 |
| 13 | ` R` | 0 |
| 14 | `ew` | 0 |
| 15 | `rite` | 0 |
| 16 | ` the` | 0 |
| 17 | ` input` | 0 |
| 18 | ` text` | 0 |
| 19 | ` to` | 0 |
| 20 | ` make` | 0 |
| 21 | ` it` | 0 |
| 22 | ` more` | 0 |
| 23 | ` professional` | 0 |
| 24 | ` and` | 0 |
| 25 | ` formal` | 0 |
| 26 | ` while` | 0 |
| 27 | ` retaining` | 0 |
| 28 | ` its` | 0 |
| 29 | ` essential` | 0 |
| 30 | ` content` | 0 |
| 31 | `.\n\n` | 0 |
| 32 | `Hey` | 0 |
| 33 | ` Alex` | 0 |
| 34 | `,\n\n` | 0 |
| 35 | `So` | 0 |
| 36 | `,` | 0 |
| 37 | ` I` | 0 |
| 38 | `'ve` | 0 |
| 39 | ` been` | 0 |
| 40 | ` thinking` | 0 |
| 41 | ` a` | 0 |
| 42 | ` lot` | 0 |
| 43 | ` about` | 0 |
| 44 | ` our` | 0 |
| 45 | ` new` | 0 |
| 46 | ` project` | 0 |
| 47 | ` on` | 0 |
| 48 | ` the` | 0 |
| 49 | ` e` | 0 |
| 50 | `tym` | 0 |
| 51 | `ology` | 0 |
| 52 | ` of` | 0 |
| 53 | ` ancient` | 0 |
| 54 | ` languages` | 0 |
| 55 | `,` | 0 |
| 56 | ` and` | 0 |
| 57 | ` I` | 0 |
| 58 | ` have` | 0 |
| 59 | ` a` | 0 |
| 60 | ` ton` | 0 |
| ... | (296 more tokens) | |

### MMLU auxiliary_train - 237 tokens, 2 supervised (0.8%)

| pos | token | mask |
|---|---|---|
| 0 | `<\|bos\|>` | 0 |
| 1 | `<\|user_start\|>` | 0 |
| 2 | `Multiple` | 0 |
| 3 | ` Choice` | 0 |
| 4 | ` question` | 0 |
| 5 | `:` | 0 |
| 6 | ` Rules` | 0 |
| 7 | ` in` | 0 |
| 8 | ` the` | 0 |
| 9 | ` reading` | 0 |
| 10 | ` room` | 0 |
| 11 | ` Hello` | 0 |
| 12 | `,` | 0 |
| 13 | ` everyone` | 0 |
| 14 | `.` | 0 |
| 15 | ` Welcome` | 0 |
| 16 | ` to` | 0 |
| 17 | ` the` | 0 |
| 18 | ` school` | 0 |
| 19 | ` reading` | 0 |
| 20 | ` room` | 0 |
| 21 | `.` | 0 |
| 22 | ` We` | 0 |
| 23 | ` hope` | 0 |
| 24 | ` you` | 0 |
| 25 | ` have` | 0 |
| 26 | ` a` | 0 |
| 27 | ` good` | 0 |
| 28 | ` time` | 0 |
| 29 | ` here` | 0 |
| 30 | `.` | 0 |
| 31 | ` Before` | 0 |
| 32 | ` you` | 0 |
| 33 | ` go` | 0 |
| 34 | ` into` | 0 |
| 35 | ` the` | 0 |
| 36 | ` reading` | 0 |
| 37 | ` room` | 0 |
| 38 | `,` | 0 |
| 39 | ` there` | 0 |
| 40 | ` are` | 0 |
| 41 | ` some` | 0 |
| 42 | ` rules` | 0 |
| 43 | ` you` | 0 |
| 44 | ` need` | 0 |
| 45 | ` to` | 0 |
| 46 | ` keep` | 0 |
| 47 | `.` | 0 |
| 48 | ` ` | 0 |
| 49 | `1` | 0 |
| 50 | `.The` | 0 |
| 51 | ` reading` | 0 |
| 52 | ` room` | 0 |
| 53 | ` is` | 0 |
| 54 | ` open` | 0 |
| 55 | ` from` | 0 |
| 56 | ` ` | 0 |
| 57 | `8` | 0 |
| 58 | `:` | 0 |
| 59 | `00` | 0 |
| 60 | ` a` | 0 |
| ... | (176 more tokens) | |

### GSM8K train - 129 tokens, 61 supervised (47.3%)

| pos | token | mask |
|---|---|---|
| 0 | `<\|bos\|>` | 0 |
| 1 | `<\|user_start\|>` | 0 |
| 2 | `M` | 0 |
| 3 | `imi` | 0 |
| 4 | ` picked` | 0 |
| 5 | ` up` | 0 |
| 6 | ` ` | 0 |
| 7 | `2` | 0 |
| 8 | ` dozen` | 0 |
| 9 | ` seas` | 0 |
| 10 | `hells` | 0 |
| 11 | ` on` | 0 |
| 12 | ` the` | 0 |
| 13 | ` beach` | 0 |
| 14 | `.` | 0 |
| 15 | ` ` | 0 |
| 16 | ` K` | 0 |
| 17 | `yle` | 0 |
| 18 | ` found` | 0 |
| 19 | ` twice` | 0 |
| 20 | ` as` | 0 |
| 21 | ` many` | 0 |
| 22 | ` shells` | 0 |
| 23 | ` as` | 0 |
| 24 | ` M` | 0 |
| 25 | `imi` | 0 |
| 26 | ` and` | 0 |
| 27 | ` put` | 0 |
| 28 | ` them` | 0 |
| 29 | ` in` | 0 |
| 30 | ` his` | 0 |
| 31 | ` pocket` | 0 |
| 32 | `.` | 0 |
| 33 | ` Le` | 0 |
| 34 | `igh` | 0 |
| 35 | ` grab` | 0 |
| 36 | `bed` | 0 |
| 37 | ` one` | 0 |
| 38 | `-third` | 0 |
| 39 | ` of` | 0 |
| 40 | ` the` | 0 |
| 41 | ` shells` | 0 |
| 42 | ` that` | 0 |
| 43 | ` K` | 0 |
| 44 | `yle` | 0 |
| 45 | ` found` | 0 |
| 46 | `.` | 0 |
| 47 | ` ` | 0 |
| 48 | ` How` | 0 |
| 49 | ` many` | 0 |
| 50 | ` seas` | 0 |
| 51 | `hells` | 0 |
| 52 | ` did` | 0 |
| 53 | ` Le` | 0 |
| 54 | `igh` | 0 |
| 55 | ` have` | 0 |
| 56 | `?` | 0 |
| 57 | `<\|user_end\|>` | 0 |
| 58 | `<\|assistant_start\|>` | 0 |
| 59 | `M` | 1 |
| 60 | `imi` | 1 |
| ... | (68 more tokens) | |


## Token statistics (sample of 2000 examples per dataset)

| dataset | mean tokens/conv | median | max | mean supervised share | est. tokens per epoch |
|---|---|---|---|---|---|
| SmolTalk train | 792 | 858 | 2048 | 73.1% | 364.4M |
| MMLU auxiliary_train | 383 | 404 | 1169 | 0.8% | 38.3M |
| GSM8K train | 174 | 164 | 427 | 56.4% | 1.3M |

Estimated tokens seen: stage 1 (mid) ~120M, stage 2 (sft) ~364M (before packing/padding).
