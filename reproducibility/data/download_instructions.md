# Dataset Download Instructions

## ShareGPT

Source: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
File used: ShareGPT_V3_unfiltered_cleaned_split.json
Used for: prompt and generation length distributions in the vLLM throughput benchmark and the discrete event simulator request generator.
Preprocessing: filtered to usable single or multi turn text, tokenized with the TinyLlama tokenizer, empty or oversized turns removed.
Split: no train or test split, used purely as a workload trace.

## Azure LLM Inference Trace (Conversation)

Source: https://github.com/Azure/AzurePublicDataset, file data/AzureLLMInferenceTrace_conv.csv
Attribution: Patel, Choukse, Zhang, Shah, Goiri, Maleki, and Bianchini. Splitwise: Efficient Generative LLM Inference Using Phase Splitting. ISCA 2024.
License: CC BY 4.0
Used for: arrival pattern and length distribution source for the mixed length, bursty cloud workload scenario.