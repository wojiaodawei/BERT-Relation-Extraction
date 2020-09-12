from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import logging

import torch
from torch import nn
from transformers import BertPreTrainedModel
from transformers.modeling_bert import (
    BertEmbeddings,
    BertEncoder,
    BertOnlyMLMHead,
    BertPooler,
)

logger = logging.getLogger(__name__)

_TOKENIZER_FOR_DOC = "BertTokenizer"


class BertModel(BertPreTrainedModel):
    def __init__(self, config, model_size, task=None, n_classes=None):
        """
        The bare Bert Model transformer outputting raw hidden-states without
        any specific head on top.

        The model can behave as an encoder (with only self-attention) as well as a
        decoder, in which case a layer of cross-attention is added between the
        self-attention layers, following the architecture described in `Attention
        is all you need`_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
        Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

        This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
        Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
        usage and behavior.

        Args:
            config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
                Initializing with a config file does not load the weights associated with the model, only the configuration.
                Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
            model_size: Size of the Model
            task: MTB task
            n_classes: Number of classes

        References:
            Attention is all you need (https://arxiv.org/abs/1706.03762)
        """
        super(BertModel, self).__init__(config)
        self.config = config

        self.task = task
        self.model_size = model_size
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

        logger.info("Model config: ", self.config)
        if self.task is None:
            self.lm_head = BertOnlyMLMHead(config)
        elif self.task == "classification":
            self.n_classes = n_classes
            if self.model_size == "bert-base-uncased":
                self.classification_layer = nn.Linear(1536, n_classes)
            elif self.model_size == "bert-large-uncased":
                self.classification_layer = nn.Linear(2048, n_classes)

    def get_input_embeddings(self):  # noqa: D102
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):  # noqa: D102
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model.

        See base class PreTrainedModel

        Args:
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        encoder_layer = self.encoder.layer
        for layer, heads in heads_to_prune.items():
            encoder_layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.FloatTensor = None,
        token_type_ids: torch.LongTensor = None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask: torch.FloatTensor = None,
        output_attentions=None,
        output_hidden_states=None,
        e1_e2_start=None,
    ):
        """
        Forward pass of BERT.

        Args:
            input_ids: Indices of input sequence tokens in the vocabulary.
                Indices can be obtained using transformers.BertTokenizer.
            attention_mask: Mask to avoid performing attention on padding token
                indices. Mask values selected in [0, 1]: 1 for tokens that are
                NOT MASKED, 0 for MASKED tokens.
            token_type_ids: Segment token indices to indicate first and second
                portions of the inputs. Indices are selected in [0, 1]: 0
                corresponds to a `sentence A` token, 1 corresponds to a
                `sentence B` token.
            position_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
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
            encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                if the model is configured as a decoder.
            encoder_attention_mask: Mask to avoid performing attention on the padding token indices of the encoder input.
                This mask is used in the cross-attention if the model is configured as a decoder.
                Mask values selected in [0, 1]: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
            output_attentions (:obj:`bool`, `optional`, defaults to :obj:`None`):
                If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under returned tensors for more detail.
            output_hidden_states: Output_hidden state
            e1_e2_start: Start of entity1 and entity2 markers.

        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the model.
            pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
                Last layer hidden-state of the first token of the sequence (classification token)
                further processed by a Linear layer and a Tanh activation function. The Linear
                layer weights are trained from the next sentence prediction (classification)
                objective during pre-training. This output is usually *not* a good summary
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

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = (
            self.get_extended_attention_mask(
                attention_mask, input_shape, device
            )
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size,
                encoder_sequence_length,
            )
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device
                )
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers
        )

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        self.pooler(sequence_output)

        blanks_entity_start_hidden = sequence_output[:, e1_e2_start, :]
        buffer = []
        for i in range(blanks_entity_start_hidden.shape[0]):
            e1e2_merged = blanks_entity_start_hidden[i, i, :, :]
            e1e2_merged = torch.cat((e1e2_merged[0], e1e2_merged[1]))
            buffer.append(e1e2_merged)
        e1e2_merged = torch.stack(list(buffer), dim=0)

        if self.task is None:
            blanks_logits = e1e2_merged
            lm_logits = self.lm_head(sequence_output)
            return blanks_logits, lm_logits
        elif self.task == "classification":
            size = 1536 if self.model_size == "bert-base-uncased" else 2048
            normalized_v1v2 = torch.nn.LayerNorm(
                size, elementwise_affine=False
            )(e1e2_merged)
            classification_logits = self.classification_layer(normalized_v1v2)
            return torch.nn.Softmax(1)(classification_logits)
        elif self.task == "fewrel":
            return e1e2_merged
