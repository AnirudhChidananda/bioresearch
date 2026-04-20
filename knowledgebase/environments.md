# Bioresearch Environment

A biological reasoning environment for training and evaluating AI agents on real-world genomics and proteomics tasks. The environment presents agents with DNA mutation analysis and protein function prediction challenges, grading responses with domain-specific scoring that rewards both accuracy and scientific reasoning quality.

## Motivation

Understanding genetic variants and protein function is central to modern biomedical research and of huge value to the **_drug discovery_** process. This environment evaluates whether AI agents can:

- Classify the pathogenic effects of DNA mutations using pathway and sequence context
- Articulate high quality biological reasoning traces explaining mutation-to-disease mechanisms
- Predict protein function, subcellular localisation, and Gene Ontology annotations from sequence data

These are tasks that human experts routinely perform, making this a genuine real-world evaluation benchmark.

## Tasks

The environment provides four tasks of increasing difficulty:

DNA Mutation Effect Predictor - The agent receives a DNA sequence and a proposed mutation. - The reward signal comes from correctly predicting whether the variant is benign or pathogenic (like BioReason's BRCA2 example). - GRPO can be used to train reasoning traces that explain why a mutation is harmful.

Reasoning biological model - The goal is to connect this model connects Arc's Evo 2 (genomic foundation model) with a reasoning language model. It uses reinforcement learning to teach the model to reason about DNA sequences. - 80% of the effort goes into data curation. They use a human-in-the-loop approach where frontier models (like GPT-5) generate reasoning traces, which are then verified and improved by human experts - They utilize reinforcement learning (specifically GRPO) to fine-tune the models on these high-quality, step-by-step reasoning traces.

Protein Function Hypothesis Generator - The goal is to addresses the sequence-to-function gap. While AlphaFold solved structure prediction, We aim to annotate protein functions from sequences. - Annotating protein functions from sequences involves predicting biological roles, structural domains, and molecular characteristics using computational tools - the model can sometimes go deeper into biological trees than existing curated databases (like Gene Ontology terms), functioning as a useful hypothesis generator. - Given an unannotated protein sequence, the agent must generate a ranked list of functional hypotheses with supporting reasoning (domain homology, motif presence, evolutionary context). Evaluated against expert annotations or literature.

Single-Cell Perturbation Predictor — Inspired by BioReason 3's goal. The agent receives a virtual cell state (gene expression profile) and a proposed perturbation (gene knockout, drug treatment). It must predict the resulting expression changes and reason about the causal pathway. Reward from Perturb-seq datasets.
