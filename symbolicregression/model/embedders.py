# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
from symbolicregression.utils import to_cuda
import torch.nn.functional as F

MultiDimensionalFloat = List[float]
XYPair = Tuple[MultiDimensionalFloat, MultiDimensionalFloat]
Sequence = List[XYPair]

    
class Embedder(ABC, nn.Module):
    """
    Base class for embedders, transforms a sequence of pairs into a sequence of embeddings.
    """

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def forward(self, sequences: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def encode(self, sequences: List[Sequence]) -> List[torch.Tensor]:
        pass

    def batch(self, seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_length_after_batching(self, sequences: List[Sequence]) -> List[int]:
        pass

class LinearPointEmbedder(Embedder):
    def __init__(self, params, env):
        from .transformer import Embedding

        super().__init__()
        self.env = env
        self.params = params
        self.input_dim = params.emb_emb_dim
        self.output_dim = params.enc_emb_dim
        self.embeddings = Embedding(
            len(self.env.float_id2word),
            self.input_dim,
            padding_idx=self.env.float_word2id["<PAD>"],
        )
        self.float_scalar_descriptor_len = (2 + self.params.mantissa_len)
        self.total_dimension = 1 + self.params.max_dimension
        self.float_vector_descriptor_len = self.float_scalar_descriptor_len * self.total_dimension

        self.activation_fn = getattr(F, params.activation)
        size = self.float_vector_descriptor_len*self.input_dim
        hidden_size = size * self.params.emb_expansion_factor
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(size, hidden_size))
        for i in range(self.params.n_emb_layers-1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.fc = nn.Linear(hidden_size, self.output_dim)
        self.max_seq_len = self.params.max_points

    def forward(self, sequences: List[Sequence], return_before_embed=False) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = self.encode(sequences)
        sequences, sequences_len = self.batch(sequences)
        sequences, sequences_len = to_cuda(sequences, sequences_len, use_cpu=self.fc.weight.device.type=="cpu")
        if return_before_embed:
            return sequences, sequences_len
        sequences_embeddings = self.embed(sequences)
        sequences_embeddings = self.compress(sequences_embeddings)
        return sequences_embeddings, sequences_len

    def compress(
        self, sequences_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes: (N_max * (d_in+d_out)*(2+mantissa_len), B, d) tensors
        Returns: (N_max, B, d)
        """
        max_len, bs, float_descriptor_length, dim = sequences_embeddings.size()
        sequences_embeddings = sequences_embeddings.view(max_len, bs, -1)
        for layer in self.hidden_layers: sequences_embeddings = self.activation_fn(layer(sequences_embeddings))
        sequences_embeddings = self.fc(sequences_embeddings)
        return sequences_embeddings
    
    def encode(self, sequences: List[Sequence], pairs=True) -> List[torch.Tensor]:
        res = []
        for seq in sequences:
            seq_toks = []
            for x, y in seq:
                x_toks = self.env.float_encoder.encode(x)
                y_toks = self.env.float_encoder.encode(y)
                input_dim = int(len(x_toks) / (2 + self.params.mantissa_len))
                output_dim = int(len(y_toks) / (2 + self.params.mantissa_len))
                x_toks = [
                    *x_toks,
                ]
                y_toks = [
                    *y_toks,
                    *[
                        "<OUTPUT_PAD>"
                        for _ in range(
                            (self.params.max_dimension - output_dim)
                            * self.float_scalar_descriptor_len
                        )
                    ],
                ]
                toks = [*x_toks, *y_toks]
                seq_toks.append([self.env.float_word2id[tok] for tok in toks])
            res.append(torch.LongTensor(seq_toks))
        return res

    def batch(self, seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_id = self.env.float_word2id["<PAD>"]
        lengths = [len(x) for x in seqs]
        bs, slen = len(lengths), max(lengths)
        sent = torch.LongTensor(slen, bs, self.float_vector_descriptor_len).fill_(pad_id)
        for i, seq in enumerate(seqs):
            sent[0 : len(seq), i, :] = seq
        return sent, torch.LongTensor(lengths)

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        return self.embeddings(batch)

    def get_length_after_batching(self, seqs: List[Sequence]) -> torch.Tensor:
        lengths = torch.zeros(len(seqs), dtype=torch.long)
        for i, seq in enumerate(seqs):
            lengths[i] = len(seq)
        assert lengths.max() <= self.max_seq_len, "issue with lengths after batching"
        return lengths
    

class TwoHotEmbedder():
    # TODO: support masked-language-modeling
    # TODO: check if compatible with old two-hot-encoding
    # TODO: add scaling function
    def __init__(
        self, 
        num_embeddings: Union[None, int], # if None, you must supply min_value and max_value
        embedding_dim: int,
        min_value: Union[None, int] = None, # min is included, ignored if num_embeddings is not None
        max_value: Union[None, int] = None, # min is excluded, ignored if num_embeddings is not None
        init_from_arange: bool = False, # useful for debugging
    ):
        """
        Arguments:
        ----------
        - num_embeddings: number of embeddings.
        - embedding_dim: dimension of embeddings.
        - min_value: smallest value to be represented (min is included).
        - max_value: supremum, this is the first value outside the represented range of values.
        -init_from_arange: If True, initialize embeddings such that an input of 0 gets an embedding vector of zeros, 
            an input of 1 gets an embedding vector of ones, etc.
        
        """
        self.max_dimension = 1 #TODO
        self.float_scalar_descriptor_len = 1
        self.total_dimension = 1 + self.max_dimension
        self.float_vector_descriptor_len = self.float_scalar_descriptor_len * self.total_dimension
        self.embedding_dim = embedding_dim
        self._min_value = min_value
        self._max_value = max_value
        if num_embeddings is None:
            assert max_value > min_value, \
                f"max_value must be larger than min_value but found {max_value} vs {min_value}."
            num_embeddings = max_value - min_value + 2
        elif min_value is not None or max_value is not None:
            print("min_value and max_value will be ignored since num_embeddings is not None.")
        self.num_embeddings = num_embeddings
        self.pad_embedding_value = self.num_embeddings
        self.init_from_arange = init_from_arange
        if init_from_arange:
            self.embd_bag = torch.nn.EmbeddingBag.from_pretrained(
                embeddings=torch.arange(
                    min_value, max_value+2, dtype=torch.float32
                ).reshape(-1, 1).repeat(1, self.embedding_dim),
                padding_idx=self.num_embeddings-1,
                mode="sum",
            )
        else:
            self.embd_bag = torch.nn.EmbeddingBag(
                num_embeddings=self.num_embeddings,
                embedding_dim=self.embedding_dim,
                padding_idx=self.num_embeddings-1,
                mode="sum",
            )
    
    def __call__(self, inputs: List[Sequence]) -> Tuple[torch.Tensor, torch.LongTensor]:
        return self.forward(inputs)
    
    def forward(self, inputs: List[Sequence]) -> Tuple[torch.Tensor, torch.LongTensor]:
        inputs = self.encode(inputs)
        if isinstance(inputs, List):
            inputs, sequences_len = self.batch(inputs)
        else:
            sequences_len = torch.LongTensor(
                [inputs.shape[0]] * (1 if len(inputs.shape) == 1 else inputs.shape[1])
            )
        return self.embed(inputs), sequences_len
        
    
    def encode(self, sequences = List[Sequence]) -> List[torch.Tensor]:
        res = []
        for seq in sequences:
            seq_toks = []
            for x_toks, y_toks in seq:
                if isinstance(x_toks, (float, int)):
                    x_toks = [x_toks]
                system_dim = len(y_toks)
                x_toks = [*x_toks,]
                y_toks = [
                    *y_toks,
                    *[
                        self.embd_bag.padding_idx
                        for _ in range(
                            (self.max_dimension - system_dim)
                            * self.float_scalar_descriptor_len
                        )
                    ],
                ]
                seq_toks.append([*x_toks, *y_toks])
            res.append(torch.DoubleTensor(seq_toks))
        return res
    
    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, system_dims = inputs.shape
        # for the two-hot representation we need to specify:
        # - the neighboring bins that support the distribution (=support_idcs)
        # - the probability mass that is assigned to each of the bins (=support_weights)
        support_idcs = torch.stack((inputs.reshape(-1), inputs.reshape(-1))).T
        # right bins contains the decimal value, e.g. 0.6 for input 1.6
        support_weights = support_idcs % 1
        # left bin contains 1 minus decimal value, e.g. 0.4 for input 1.6
        support_weights[:, 0] = 1 - support_weights[:, 1]
        support_weights = support_weights.to(torch.float32)
        support_idcs[:, 0] = torch.floor(support_idcs[:, 0])
        support_idcs[:, 1] = torch.ceil(support_idcs[:, 1])
        support_idcs = support_idcs.to(torch.int64)
        support_idcs[support_idcs != self.embd_bag.padding_idx] -= self._min_value
        embedding = self.embd_bag(
            input=support_idcs, per_sample_weights=support_weights,
        )
        if self.init_from_arange:
            embedding[support_idcs[:,0]==self.embd_bag.padding_idx] = self.pad_embedding_value
        # concat embeddings per dimension
        return embedding.reshape(seq_len, batch_size, -1)
        
    def batch(self, seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.LongTensor]:
        pad_id = self.embd_bag.padding_idx
        lengths = [len(x) for x in seqs]
        bs, slen = len(lengths), max(lengths)
        sent = torch.DoubleTensor(slen, bs, self.float_vector_descriptor_len).fill_(pad_id)
        for i, seq in enumerate(seqs):
            sent[0 : len(seq), i, :] = seq
        return sent, torch.LongTensor(lengths)
    
    
