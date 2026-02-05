# Covid Tweet Trigram Language Model

# Overview
This project implements a statistical trigram language model trained on a large corpus of COVID-19–related tweets from 2020. The system processes raw social media text, learns word sequence probabilities, and uses those probabilities to analyze and generate language.

The focus of this project is data-driven NLP: transforming text into structured statistics and validating correctness through in-code test cases.

# Tech Stack and Core Concepts
Language: Python

Core Concepts:
- N-gram modeling (trigrams)
- Probability-based next-word prediction
- Beam search for sentence generation
- Sampling from probability distributions
- Efficient dictionary-based data structures for large datasets

# Project Structure

covidtweettrigram.py — Main Python script implementing trigram language model and text generation

NgramLM class
- Stores trigrams in dictionaries for fast look-up
- Computes top next words and samples next words based on frequency counts
- Generates sentences using beam search or probabilistic sampling

Test Cases
- Demonstrates model functionality with example prefixes such as <BOS1> <BOS2> trump or <BOS1> <BOS2> biden
- Evaluates top next-word predictions, sampled predictions, and beam search generation

Data
- data/tweets/covid-tweets-2020-08-10-2020-08-21.trigrams.txt — Trigram counts extracted from COVID-19 tweets

# Learning Outcomes
- Implemented a trigram language model for large-scale tweet data
- Gained hands-on experience with text generation and probabilistic modeling
- Applied beam search to generate coherent sequences from n-gram probabilities
- Strengthened understanding of efficient dictionary-based data structures in Python
