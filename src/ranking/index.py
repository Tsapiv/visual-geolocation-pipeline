from typing import Optional

import numpy as np
import torch
from torch.nn.functional import cosine_similarity

from dataset import Dataset
from ranking.gnn_reranking import gnn_reranking


class Index:

    def __init__(self, dataset: Dataset, device: Optional[torch.device] = None):
        self.__dataset = dataset
        self.__base_descriptors: Optional[torch.FloatTensor] = None
        self.__uuids: np.ndarray[str] = np.asarray(self.__dataset.entries)
        self.__device = device if device is not None else (torch.device('cuda')
                                                           if torch.cuda.is_available() else torch.device('cpu'))

    def topk(self, query_descriptors: np.ndarray, k: int, rerank: bool = True):
        if len(query_descriptors.shape) == 1:
            query_descriptors = query_descriptors.reshape(1, -1)
        query_descriptors /= np.linalg.norm(query_descriptors, axis=-1, keepdims=True)
        query_descriptors = torch.FloatTensor(query_descriptors).to(self.__device)
        if self.__base_descriptors is None:
            base_descriptors = np.asarray(list(self.__dataset.descriptor(self.__uuids)))
            base_descriptors /= np.linalg.norm(base_descriptors, axis=-1, keepdims=True)
            self.__base_descriptors = torch.FloatTensor(base_descriptors).to(self.__device)

        if not rerank:
            similarities = cosine_similarity(self.__base_descriptors, query_descriptors)
            S, ranking = similarities.topk(k=k, dim=-1, largest=True, sorted=True)
            return self.__uuids[np.squeeze(ranking.cpu().numpy())]
        else:
            k1 = 3 * k
            k2 = k
            ranking = np.squeeze(gnn_reranking(query_descriptors, self.__base_descriptors, k1, k2))[:k2]
            return self.__uuids[ranking]
