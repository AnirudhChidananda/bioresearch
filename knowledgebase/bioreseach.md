Understanding BioReason
The discussion begins on the BioReason model series, which applies reasoning language models to biology.
BioReason 1: Bo explains that this model connects Arc's Evo 2 (genomic foundation model) with Qwen 3 (a reasoning language model). It uses reinforcement learning to teach the model to reason about DNA sequences.
Hani discusses the synergy between biological foundation models (which handle biological abstractions) and reasoning models (which possess world knowledge and stepwise thinking).

Data Curation & Training
Bo and Hani explain that 80% of their effort goes into data curation. They use a human-in-the-loop approach where frontier models (like GPT-5) generate reasoning traces, which are then verified and improved by human experts.
They utilize reinforcement learning (specifically GRPO) to fine-tune the models on these high-quality, step-by-step reasoning traces.

BioReason-Pro (BioReason 2)
BioReason-Pro addresses the sequence-to-function gap. While AlphaFold solved structure prediction, BioReason-Pro aims to annotate protein functions from sequences.
Hani notes the model can sometimes go deeper into biological trees than existing curated databases (like GO terms), functioning as a useful hypothesis generator.
They discuss current limitations, such as sensitivity to single amino acid changes, noting that further training with specific variant effect data will improve this.

The Future of BioReason
They discuss BioReason 3, which aims to interpret cellular/virtual cell models and single-cell data.
The ultimate goal is creating multiscale models that can bridge the gap from DNA/protein mutations to cellular responses and, eventually, clinical applications.
Hani and Bo emphasize that these models will revolutionize biomedical research by enabling automated literature surveys, agentic frameworks, and collective intelligence systems.

#Video summary
This video features a conversation with Dr. Hani Goodarzi (Arc Institute) and Dr. Bo Wang (Xaira Therapeutics) regarding BioReason, an open-source series of models designed to bring human-like reasoning to biological sequence analysis. The discussion highlights the shift from standard prediction models to systems that can explain their biological conclusions via reasoning traces.

Key Highlights
BioReason 1: DNA Reasoning: This model integrates Arc Institute’s Evo 2 (genomic foundation model) with Qwen 3 (a reasoning language model). By training on DNA sequences using reinforcement learning, the model learns to "think" about biological modalities in a way that allows it to interpret mutations and their downstream effects.
The Synergy of Models: The team emphasizes the importance of combining the strengths of biological foundation models (which handle complex data abstractions) with reasoning models (which possess vast world knowledge and stepwise logical processes). This combination emulates how human experts synthesize biological data.
Data Curation & Training: Approximately 80% of the project effort is dedicated to data curation. The team employs a "human-in-the-loop" process, using frontier models to generate reasoning traces, which are then rigorously verified and refined by expert biologists. This high-quality data is used to fine-tune the models via GRPO (Group Relative Policy Optimization).
BioReason-Pro : This model extends the reasoning framework to protein functions. While structural tools like AlphaFold are excellent for determining the shape of a protein, BioReason-Pro attempts to bridge the critical sequence-to-function gap, enabling researchers to hypothesize the roles of unannotated proteins.
Current Limitations & Future Outlook: The models currently face challenges with sensitivity to single amino acid mutations and occasional hallucinations. The team is already working on BioReason 3, which will focus on virtual cell models and single-cell data, with the long-term goal of building multi-scale systems that connect molecular mutations to patient-level clinical outcomes.
Applications & Impact
The speakers note that these models act as powerful hypothesis generators rather than just prediction machines. By providing human-readable reasoning, they empower researchers to verify AI outputs, potentially revolutionizing biomedical research through automated literature surveys, agentic workflows, and the eventual development of collective intelligence systems.
