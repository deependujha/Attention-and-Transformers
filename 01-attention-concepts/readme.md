# Concept of Self-Attention

## Introduction

- Most important part in any nlp product is the embedding of the words.
- But, the embedding of a word we use remains same throughout the sentence (**irrespective of context**).
- This poses a problem.
- For example,
```
sentence 1: An Apple a day keeps the doctor away.
sentence 2: Apple just launched a new iPhone.

=================================

- In normal embedding, the word "Apple" will have same embedding in both the sentences.

- But, in reality, the word "Apple" in sentence 1 is a fruit and in sentence 2 is a company.
```

- This is where **self-attention** comes into picture.

---

## What is Self-Attention?

![self attention campusx](../assets/self-attention-campusx.png)

- We pass in `normal embeddings` of words into **self-attention** and it gives us `contextual embeddings` of words.

- it helps to `capture dependencies` and `relationships within input sequences`.

- It allows the model to identify and weigh the importance of different parts of the input sequence by attending to itself.

---


