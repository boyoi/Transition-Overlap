
import random

import numpy as np
import torch

from benchmarks_ours.data_sets import wikimqa, musique, samsum, multi_news, hotpotqa, needle, passage_retrieval_en, long_context, dummy


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Utils for datasets
str2class = {
    '2wikimqa': wikimqa.WikiMQA,
    'musique': musique.Musique,
    'multi_news': multi_news.MultiNews,
    'hotpotqa': hotpotqa.HotpotQA,
    'samsum': samsum.SAMSum,
    '2wikimqa-beginning': wikimqa.WikiMQA,
    '2wikimqa-middle': wikimqa.WikiMQA,
    '2wikimqa-end': wikimqa.WikiMQA,
    '2wikimqa-original': wikimqa.WikiMQA, 
    'needle': needle.Needle,   
    'passage_retrieval_en': passage_retrieval_en.PassageRetrievalEn,
    'long_context': long_context.LongContext,
    'dummy': dummy.Dummy,
}


# Utils for plottings
model_names_map = {
        'Qwen/Qwen3-4B': 'Qwen3 4B',
        'mistralai/Mistral-7B-Instruct-v0.2': 'Mistral 7B Instruct',
        'meta-llama/Llama-3.1-70B-Instruct': 'Llama 3.1 70B Instruct',
}
dataset_names_map = {
    '2wikimqa': '2WikiMQA',
    'musique': 'MuSiQue',
    'samsum': 'SAMSum',
    'multi_news': 'MultiNews',
    'hotpotqa': 'HotpotQA',
    'needle': 'Needle'
}
dataset_metrics_map = {
    '2wikimqa': 'F1 score',
    'musique': 'F1 score',
    'samsum': 'Rouge-L score',
    'multi_news': 'Rouge-L score',
    'hotpotqa': 'F1 score',
    'needle': 'F1 score',
}

approach_names_map = {
    'fr': 'FR',
    'naive': 'Naive',
    'cacheblend-20': 'CacheBlend-20',
    'cacheblend-15': 'CacheBlend-15',
    'cacheblend-10': 'CacheBlend-10',
    'cacheblend-5': 'CacheBlend-5',
    'cacheblend-1': 'CacheBlend-1',
    'epic-32': 'EPIC-32',
    'epic-16': 'EPIC-16',
    'epic-8': 'EPIC-8',
    'epic-4': 'EPIC-4',
    'epic-2': 'EPIC-2',
    'epic-1': 'EPIC-1',
    'epic-0': 'EPIC-0',
    'CSO': 'CSO',
    'SSO': 'SSO',
}
