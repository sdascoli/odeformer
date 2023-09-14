# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from copy import copy
import os
import torch
from .embedders import LinearPointEmbedder, TwoHotEmbedder
from .transformer import TransformerModel
from .sklearn_wrapper import SymbolicTransformerRegressor
from .model_wrapper import ModelWrapper
from .mixins import (
    BatchMixin, FiniteDifferenceMixin, PredictionIntegrationMixin
)
import torch.nn as nn

logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    if params.enc_emb_dim is None:
        params.enc_emb_dim = params.emb_emb_dim
    if params.dec_emb_dim is None:
        params.dec_emb_dim = params.emb_emb_dim
    # model dimensions
    assert params.enc_emb_dim % params.n_enc_heads == 0
    assert params.dec_emb_dim % params.n_dec_heads == 0

    # reload a pretrained model
    if params.reload_model != "":
        print("Reloading model from ", params.reload_model)
        assert os.path.isfile(params.reload_model)


def build_modules(env, params):
    """
    Build modules.
    """
    modules = {}
    modules["embedder"] = LinearPointEmbedder(params, env)
    env.get_length_after_batching = modules["embedder"].get_length_after_batching

    modules["encoder"] = TransformerModel(
        params,
        env.float_id2word,
        is_encoder=True,
        with_output=False,
        use_prior_embeddings=True,
        positional_embeddings=params.enc_positional_embeddings,
    )
    if params.use_two_hot:
        dec_id2word = copy(env.equation_id2word)
        dec_id2word.update(env.constant_id2word)
    else:
        dec_id2word = env.equation_id2word
    if params.masked_input:
        modules["encoder"].proj = nn.Linear(params.enc_emb_dim, 
                                            self.params.float_descriptor_length*params.max_dimension*len(env.float_id2word), 
                                            bias=True) 

    modules["decoder"] = TransformerModel(
        params,
        dec_id2word,
        is_encoder=False,
        with_output=True,
        use_prior_embeddings=False,
        positional_embeddings=params.dec_positional_embeddings,
    )

    # reload pretrained modules
    if params.reload_model != "":
        logger.info(f"Reloading modules from {params.reload_model} ...")
        reloaded = torch.load(params.reload_model)
        for k, v in modules.items():
            assert k in reloaded
            if all([k2.startswith("module.") for k2 in reloaded[k].keys()]):
                reloaded[k] = {
                    k2[len("module.") :]: v2 for k2, v2 in reloaded[k].items()
                }
            v.load_state_dict(reloaded[k])

    # log
    for k, v in modules.items():
        logger.debug(f"{v}: {v}")
    for k, v in modules.items():
        logger.info(
            f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if p.requires_grad])}"
        )

    # cuda
    if not params.cpu:
        for v in modules.values():
            v.cuda()

    return modules
