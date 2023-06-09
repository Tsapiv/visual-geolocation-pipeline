from typing import Optional, Union

import numpy as np
import torch
from torch.nn.functional import cosine_similarity

from .gnn_reranking import precomputed_gnn_reranking
from ..common.dataset import Dataset


class Index:

    def __init__(self, dataset: Dataset, device: Optional[torch.device] = None):
        self.__dataset = dataset
        self.__db_descriptors: Optional[torch.FloatTensor] = None
        self.__db_similarities: Optional[torch.FloatTensor] = None
        self.__uuids: np.ndarray[str] = np.asarray(self.__dataset.entries)
        self.__device = device if device is not None else (torch.device('cuda')
                                                           if torch.cuda.is_available() else torch.device('cpu'))

    def __prepare_db_descriptors(self):
        base_descriptors = np.squeeze(np.asarray(list(self.__dataset.descriptor(self.__uuids))))
        base_descriptors /= np.linalg.norm(base_descriptors, axis=-1, keepdims=True)
        self.__db_descriptors = torch.FloatTensor(base_descriptors).to(self.__device)

    def __prepare_db_similarities(self):
        self.__db_similarities = torch.mm(self.__db_descriptors, self.__db_descriptors.t())

    def __get_query_extended_similarities(self, query: torch.FloatTensor):
        left_columns = torch.mm(self.__db_descriptors, query.t())
        top_columns = left_columns.t()
        top_square = torch.mm(query, query.t())
        return torch.concatenate([
            torch.concatenate([top_square, top_columns], dim=-1),
            torch.concatenate([left_columns, self.__db_similarities], dim=-1)
        ], dim=0)

    def topk(self, query_descriptors: Union[np.ndarray, torch.Tensor], k: int, rerank: bool = True):
        if isinstance(query_descriptors, np.ndarray):
            query_descriptors = torch.FloatTensor(query_descriptors)
        query_descriptors = query_descriptors.to(self.__device)

        if len(query_descriptors.shape) == 1:
            query_descriptors = query_descriptors.reshape(1, -1)
        query_descriptors /= torch.norm(query_descriptors, dim=-1, keepdim=True)

        if self.__db_descriptors is None:
            self.__prepare_db_descriptors()

        if not rerank:
            similarities = cosine_similarity(self.__db_descriptors, query_descriptors)
            S, ranking = similarities.topk(k=k, dim=-1, largest=True, sorted=True)
            return self.__uuids[np.squeeze(ranking.cpu().numpy())]
        else:
            if self.__db_similarities is None:
                self.__prepare_db_similarities()
            k1 = 3 * k
            k2 = k
            ranking = np.squeeze(precomputed_gnn_reranking(self.__get_query_extended_similarities(query_descriptors),
                                                           query_descriptors.shape[0], k1, k2))[:k2]
            return self.__uuids[ranking]
