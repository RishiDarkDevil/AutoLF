# AutoLF
AutoLF is a framework for rapidly and efficiently labelling large-scale natural language data.

## Problem Statement
Snorkel Flow and other data programming pipelines often pose difficulty in figuring out labelling functions and make it challenging to label natural language data at scale. Here we tackle the problem of annotating large-scale data efficiently with a combination of multiple approaches to improve the existing systems available for enterprise use.
![](assets/1.png)

## Workflow Interface
_The following video demonstrates the AutoLF GUI Interface for Natural Language Data Labelling_

https://github.com/RishiDarkDevil/AutoLF/assets/52328147/e5348d1d-8d82-4f57-8270-2d2e30e103ea

To run the GUI interface you will need `streamlit`.
```console
cd interactive_labeller
streamlit run autolf.py
```

## Brief Overview
- AutoLF uses a combination of unsupervised sentence clustering and topic modelling to group similar sentences to help label a large volume of texts all at once.
- AutoLF provides advanced search features to crack down into finer and finer details of the data as well as searching for a particular concept or idea in the unlabelled corpus to help annotators have better expressivity and faster annotation capability.
- AutoLF combines Weak Supervision and Active Learning to fine-tune an ensemble of weak learners, usually small language models, to capture various views of the labelled instances in each round whose scores are combined using a Snorkel Label Model to arrive at a final label for the data.

| Topic-Sentence Clustering & Semantic Search | Weak Supervision & Active Learning  |
|---|---|
| ![](assets/2.png)  | ![](assets/3.png)  |

## Results
I tested the AutoLF workflow for tasks like Sentiment Classification and Hate Speech Classification. The following is the result of AutoLF compared to a usual Active Learning counterpart on a proprietary sentiment classification dataset.
![](assets/4.png)

**NOTE**: The Active Learning Part of the workflow is not operational in the GUI version and is tested in a Python notebook environment. The GUI needs bit of changes to incorporate the training in the GUI itself. Will be updated once ready.
