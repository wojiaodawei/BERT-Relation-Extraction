import logging

import torch
from transformers import AlbertConfig, load_tf_weights_in_albert
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_callable,
)
from transformers.modeling_albert import (
    AlbertEmbeddings,
    AlbertMLMHead,
    AlbertTransformer,
)
from transformers.modeling_utils import PreTrainedModel

logger = logging.getLogger(__name__)

_TOKENIZER_FOR_DOC = "AlbertTokenizer"


class AlbertMTBModel(PreTrainedModel):
    config_class = AlbertConfig
    base_model_prefix = "albert"

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range
            )
            is_layer = isinstance(module, (torch.nn.Linear))
            if is_layer and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class AlbertModel(AlbertMTBModel):

    config_class = AlbertConfig
    load_tf_weights = load_tf_weights_in_albert
    base_model_prefix = "albert"

    def __init__(self, config, model_size, task=None, n_classes=None):
        """
        The bare ALBERT Model transformer outputting raw hidden-states without
        any specific head on top.

        This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
        Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
        usage and behavior.

        Args:
            config (:class:`~transformers.AlbertConfig`): Model configuration class with all the parameters of the model.
                Initializing with a config file does not load the weights associated with the model, only the configuration.
                Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
            model_size: Size of the Model
            task: MTB task
            n_classes: Number of classes
        """
        super().__init__(config)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)
        self.pooler = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = torch.nn.Tanh()

        self.init_weights()

        self.task = task
        self.model_size = model_size
        if self.task is None:
            # blanks head
            self.activation = torch.nn.Tanh()
            # LM head
            self.lm_classifier = AlbertMLMHead(config)
        elif self.task == "classification":
            self.n_classes = n_classes
            if self.model_size == "albert-base-v2":
                self.classification_layer = torch.nn.Linear(1536, n_classes)
            elif self.model_size == "albert-large-v2":
                self.classification_layer = torch.nn.Linear(2048, n_classes)

    def get_input_embeddings(self):  # noqa: D102
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):  # noqa: D102
        self.embeddings.word_embeddings = value

    def _resize_token_embeddings(self, new_num_tokens):  # noqa: D102
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens
        )
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model.

        ALBERT has a different architecture in that its layers are shared across groups, which then has inner groups.
        If an ALBERT model has 12 hidden layers and 2 hidden groups, with two inner groups, there
        is a total of 4 different layers.

        These layers are flattened: the indices [0,1] correspond to the two inner groups of the first hidden layer,
        while [2,3] correspond to the two inner groups of the second hidden layer.

        Any layer with in index other than [0,1,2,3] will result in an error.
        See base class PreTrainedModel for more information about head pruning

        Args:
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        layer_groups = self.encoder.albert_layer_groups
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(
                layer - group_idx * self.config.inner_group_num
            )
            layer_groups[group_idx].albert_layers[
                inner_group_idx
            ].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        e1_e2_start=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        """
        Forward pass for the model.

        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
                Indices can be obtained using :class:`transformers.AlbertTokenizer`.
                See :func:`transformers.PreTrainedTokenizer.encode` and
                :func:`transformers.PreTrainedTokenizer` for details.
                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                Mask to avoid performing attention on padding token indices.
                Mask values selected in ``[0, 1]``:
                ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
                `What are attention masks? <../glossary.html#attention-mask>`__
            token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                Segment token indices to indicate first and second portions of the inputs.
                Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
                corresponds to a `sentence B` token
                `What are token type IDs? <../glossary.html#token-type-ids>`_
            position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                Indices of positions of each input sequence tokens in the position embeddings.
                Selected in the range ``[0, config.max_position_embeddings - 1]``.
                `What are position IDs? <../glossary.html#position-ids>`_
            head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
                Mask to nullify selected heads of the self-attention modules.
                Mask values selected in ``[0, 1]``:
                :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`, defaults to :obj:`None`):
                If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under returned tensors for more detail.

        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.AlbertConfig`) and inputs:
            last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the model.
            pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
                Last layer hidden-state of the first token of the sequence (classification token)
                further processed by a Linear layer and a Tanh activation function. The Linear
                layer weights are trained from the next sentence prediction (classification)
                objective during pre-training.

                This output is usually *not* a good summary
                of the semantic content of the input, you're often better with averaging or pooling
                the sequence of hidden-states for the whole input sequence.
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        """

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds"
            )

        device = (
            input_ids.device if input_ids is not None else inputs_embeds.device
        )

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device
            )

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers
        )

        embedding_output = self.embeddings(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = encoder_outputs[0]

        # two heads: LM and blanks
        blankv1v2 = sequence_output[:, e1_e2_start, :]
        buffer = []
        for i in range(blankv1v2.shape[0]):  # iterate batch & collect
            v1v2 = blankv1v2[i, i, :, :]
            v1v2 = torch.cat((v1v2[0], v1v2[1]))
            buffer.append(v1v2)
        v1v2 = torch.stack(list(buffer), dim=0)

        if self.task is None:
            blanks_logits = self.activation(
                v1v2
            )  # self.sigmoid(self.blanks_linear( - torch.log(Q))
            lm_logits = self.lm_classifier(sequence_output)
            return blanks_logits, lm_logits

        elif self.task == "classification":
            classification_logits = self.classification_layer(v1v2)
            return classification_logits
        elif self.task == "fewrel":
            return v1v2
