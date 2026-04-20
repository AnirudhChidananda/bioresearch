Evo 2 model Reinforcement Learning with Openenv

### Pretraining - 8K neucliotides , small context

#Potential environments

## Classify Introns and Extrons

    - Extrons are part of the gene that contains valid coding sequence
    - Introns are parts in the gene between the Extrons that doesn't code for the protien but are just there.

## Promoter Motifs

    - Marks the start of a gene in DNA Sequence
    - Like a "start reading here" sign right before the gene begins
    - Mechanistic interpretability of Evo2 reveals DNA, RNA, protien, and organism level features

### Midtraining - Train on longer sequences by giving it multiple genes at a time.

    - Analogy : It's like trying to understand a whole book by reading one page.
    - Learns different relations with the genes. Different genes might be related
    - Important to generate full sequences longer than just 8k context window.
    -

## Regulators

    - Genes aren't always turned on. They depend on Enhancers and Silencers
    - Enhancers boost genen protien production
    - Silencers decrease gene avictivty
    - They can be thousands of base pairs away (not next to each other), and be activated due to DNA Strand folding.

### Why not only train on human DNA?

    - The only human geneome is a reference genome
    - WHen given BRCA2 cancer dataset with varients/ mutations, the model could predict that the varients found in the dataset was harmful without having to be fine tuned to the BRCA2 dataset.
    - Having trained on so many other species, the broad knowledge it has gathered helps it perform better for cases like this.

### Multi Species variation

    - It learned what parts of the DNA change a lot across species and which part stay similar.
    - The parts that stay similar should be more

- Repetitive DNA consists of sequences repeated multiple times, forming a large portion of eukaryotic genomes (25-50% in mammals) and often playing roles in chromosome structure, regulation, and evolution. In contrast, non-repetitive DNA (single-copy) includes unique sequences that mostly code for proteins.

## Model info

    - Genome modeling and design accross all domains of life
    - Trained on 9.3 trillion base pairs
    - 1 million token context window
    - Knows DNA from many spicies, not just humans.
    - Architecture: It is built on the StripedHyena 2 architecture, which is a convolutional, non-Transformer framework.
    - Using Reinforcement Learning (RL) on a convolutional (CNN), non-Transformer framework is not only a good idea, but it is also the standard, proven approach for image-based or grid-world RL environments.
    - BLASTP programs search protein databases using a protein query. Enter coordinates for a subrange of the query sequence. Sequence coordinates are from 1 to the sequence length.The range includes the residue at the To coordinate.

    - the Evo 2 model developed by Arc Institute and NVIDIA significantly improves CRISPR gene editing by enabling faster, more accurate predictions of genetic mutations and designing new biological sequences at a genome scale. It acts as a foundational AI, optimizing CRISPR by predicting the functional impacts of edits, identifying harmful mutations, and accelerating the design of novel gene-editing tools

    -Synteny refers to the conserved physical co-localization of genes on the same chromosome across different species, originating from a common ancestor. Proper, high-quality synteny implies not only the presence of homologous genes in the same region but also the preservation of their linear order, which helps define evolutionary distances, genome rearrangements, and gene regulation

## Use cases

#Guided generation

    ##Designing genome for a specific goal

        - DNA is very long, but still needs to fit within a single cell.
        - The cell needs to pack the DNA very efficiently, instead of just stuffing it randomly
        - The DNA wound around a small protien (histones) spools. The spools are then wound and packed into complexes called chromatin.
        - The problem: If the genes within the DNA is packed so tightly, it is inaccessible for being read.
        - Soulution: The cell unpacks the chromatin structure around only the needed gene, making the gene accessable

        ###Chromatin accessibility : How physically and reachable the gene is within its packaged chromatin.

        ### Why chromatin accessibility is important?
        - Every cell store the full genome , but the cells in the brain won't need the same genes as the cell in the skin.
        - For example the brain keeps the "brain genes" more accesseble, while packing the "skin genes" tightly away.
        - DNA can be designed, so certain chromatins are more or less accessible.
