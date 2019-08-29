#!crnn/rnn.py
# kate: syntax python;


#
# Based on Parnia Bahar's HMM factorization paper. For the attention we average the last attention layer over the
# heads. Note, we should try different approaches here in the future.
#

import os
from Util import get_login_username
demo_name, _ = os.path.splitext(__file__)
print("Hello, experiment: %s" % demo_name)

# task
use_tensorflow = True
task = "train"
in_loop = False  # IMPORTANT: SET THIS TO TRUE IN SEARCH, FALSE IN TRAIN <-----------------------------------------------

# data
train = {"class": "TaskEpisodicCopyDataset", "num_seqs": 100}
num_inputs = 10
num_outputs = 10
#train = {"class": "TaskXmlModelingDataset", "num_seqs": 100}
#num_inputs = 12
#num_outputs = 12

dev = train.copy()
dev.update({"num_seqs": train["num_seqs"] // 10, "fixed_random_seed": 42})

# HMM Factorization data
vocab_size = 12
intermediary_size = 100


def select_random_layer_and_head(sources, in_loop):
    import tensorflow as tf

    sources = tf.stack(sources, axis=0)  # [layers, B, H, (I,) J]

    sources = tf.gather(sources, tf.random_shuffle(tf.range(tf.shape(sources)[0])))
    layer_to_use = sources[0]  # [B, H, (I,) J]

    # Select head
    if in_loop:
      layer_to_use = tf.squeeze(layer_to_use, axis=-2)
      att_perm = [1, 0, 2]
    else:
      att_perm = [1, 0, 2, 3]

    layer_to_use = tf.transpose(layer_to_use, perm=att_perm)  # [H, B, (I,) J]

    # layer_to_use = tf.Print(layer_to_use, [tf.shape(layer_to_use)], summarize=100, message="layer_to_use 2")

    layer_to_use = tf.gather(layer_to_use, tf.random_shuffle(tf.range(tf.shape(layer_to_use)[0])))
    head_to_use = layer_to_use[0]

    # head_to_use = tf.Print(head_to_use, [tf.shape(head_to_use)], summarize=100, message="head_to_use")

    head_to_use = tf.expand_dims(head_to_use, axis=-2)

    # TODO: should be then [B, (I,), 1, J]

    # head_to_use = tf.Print(head_to_use, [tf.shape(head_to_use)], summarize=100, message="head_to_use 2")

    if in_loop:
      att_perm_2 = [0, 1, 2]
    else:
      att_perm_2 = [0, 2, 1, 3]

    head_to_use = tf.transpose(head_to_use, perm=att_perm_2)  # [B, 1, I, J] or [B, 1, J]

    return head_to_use


class TransformerNetwork:

    def __init__(self, model_size=512, ff_dim=2048, encN=6, decN=6, rnn_dec=False, only_one_enc_dec_att='', normalized_loss = False):
        self.encN = encN
        self.decN = decN
        self.FFDim = ff_dim
        self.EncKeyTotalDim = model_size
        self.AttNumHeads = 8
        self.EncKeyPerHeadDim = self.EncKeyTotalDim // self.AttNumHeads
        self.EncValueTotalDim = model_size
        self.EncValuePerHeadDim = self.EncValueTotalDim // self.AttNumHeads
        self.embed_weight = self.EncValueTotalDim**0.5
        self.rnn_dec = rnn_dec
        self.only_one_enc_dec_att = only_one_enc_dec_att
        self.normalized_loss = normalized_loss

        self.embed_dropout = 0.0
        self.postprocess_dropout = 0.1
        self.act_dropout = 0.1
        self.attention_dropout = 0.1
        self.label_smoothing = 0.1

        self.ff_init = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)"

    def add_trafo_enc_layer(self, d, inp, output):
        d[output + '_self_att_laynorm'] = {"class": "layer_norm", "from": [inp]}
        d[output + '_self_att_att'] = {"class": "self_attention", "num_heads": self.AttNumHeads,
                                    "total_key_dim": self.EncKeyTotalDim,
                                    "n_out": self.EncValueTotalDim, "from": [output + '_self_att_laynorm'],
                                    "attention_left_only": False, "attention_dropout": self.attention_dropout, "forward_weights_init": self.ff_init}
        d[output + '_self_att_lin'] = {"class": "linear", "activation": None, "with_bias": False,
                                       "from": [output + '_self_att_att'], "n_out": self.EncValueTotalDim, "forward_weights_init": self.ff_init}
        d[output + '_self_att_drop'] = {"class": "dropout", "from": [output + '_self_att_lin'], "dropout": self.postprocess_dropout}
        d[output + '_self_att_out'] = {"class": "combine", "kind": "add", "from": [inp, output + '_self_att_drop'],
                                       "n_out": self.EncValueTotalDim}
        #####
        d[output + '_ff_laynorm'] = {"class": "layer_norm", "from": [output + '_self_att_out']}
        d[output + '_ff_conv1'] = {"class": "linear", "activation": "relu", "with_bias": True,
                                   "from": [output + '_ff_laynorm'],
                                   "n_out": self.FFDim, "forward_weights_init": self.ff_init}
        d[output + '_ff_conv2'] = {"class": "linear", "activation": None, "with_bias": True,
                                   "from": [output + '_ff_conv1'], "dropout": self.act_dropout,
                                   "n_out": self.EncValueTotalDim, "forward_weights_init": self.ff_init}
        d[output + '_ff_drop'] = {"class": "dropout", "from": [output + '_ff_conv2'], "dropout": self.postprocess_dropout}
        d[output + '_ff_out'] = {"class": "combine", "kind": "add",
                                 "from": [output + '_self_att_out', output + '_ff_drop'],
                                 "n_out": self.EncValueTotalDim}
        d[output] = {"class": "copy", "from": [output + '_ff_out']}

    def add_trafo_dec_layer(self, db, d, inp, output, reuse_att=None):
        pre_inp = [inp, 'prev:' + output + '_att_drop'] if self.rnn_dec else [inp]
        d[output + '_self_att_laynorm'] = {"class": "layer_norm", "from": pre_inp}
        d[output + '_self_att_att'] = {"class": "rnn_cell", "unit": "LSTMBlock",
                "from": [output + '_self_att_laynorm'], "n_out": self.EncValueTotalDim} if self.rnn_dec else {"class": "self_attention", "num_heads": self.AttNumHeads,
                                    "total_key_dim": self.EncKeyTotalDim,
                                    "n_out": self.EncValueTotalDim, "from": [output + '_self_att_laynorm'],
                                    "attention_left_only": True, "attention_dropout": self.attention_dropout, "forward_weights_init": self.ff_init}
        d[output + '_self_att_lin'] = {"class": "linear", "activation": None, "with_bias": False,
                                       "from": [output + '_self_att_att'], "n_out": self.EncValueTotalDim, "forward_weights_init": self.ff_init}
        d[output + '_self_att_drop'] = {"class": "dropout", "from": [output + '_self_att_lin'], "dropout": self.postprocess_dropout}
        d[output + '_self_att_out'] = {"class": "combine", "kind": "add", "from": [inp, output + '_self_att_drop'],
                                       "n_out": self.EncValueTotalDim}
        #####
        if not reuse_att:
            d[output + '_att_laynorm'] = {"class": "layer_norm", "from": [output + '_self_att_out']}
            d[output + '_att_query0'] = {"class": "linear", "activation": None, "with_bias": False,
                                         "from": [output + '_att_laynorm'],
                                         "n_out": self.EncValueTotalDim, "forward_weights_init": self.ff_init}
            d[output + '_att_query'] = {"class": "split_dims", "axis": "F", "dims": (self.AttNumHeads, self.EncKeyPerHeadDim),
                                        "from": [output + '_att_query0']}  # (B, H, D/H)
            db[output + '_att_key0'] = {"class": "linear", "activation": None, "with_bias": False, "from": ["encoder"],
                                        "n_out": self.EncKeyTotalDim, "forward_weights_init": self.ff_init}  # (B, enc-T, D)
            db[output + '_att_value0'] = {"class": "linear", "activation": None, "with_bias": False, "from": ["encoder"],
                                          "n_out": self.EncValueTotalDim, "forward_weights_init": self.ff_init}
            db[output + '_att_key'] = {"class": "split_dims", "axis": "F", "dims": (self.AttNumHeads, self.EncKeyPerHeadDim),
                                       "from": [output + '_att_key0']}  # (B, enc-T, H, D/H)
            db[output + '_att_value'] = {"class": "split_dims", "axis": "F", "dims": (self.AttNumHeads, self.EncValuePerHeadDim),
                                         "from": [output + '_att_value0']}  # (B, enc-T, H, D'/H)
            d[output + '_att_energy'] = {"class": "dot", "red1": -1, "red2": -1, "var1": "T", "var2": "T?",
                                         "from": ['base:' + output + '_att_key', output + '_att_query']}  # (B, H, enc-T, 1)
            d[output + '_att_weights'] = {"class": "softmax_over_spatial", "from": [output + '_att_energy'],
                                          "energy_factor": self.EncKeyPerHeadDim ** -0.5}  # (B, enc-T, H, 1)

            d[output + '_att_weights_drop'] = {"class": "dropout", "dropout_noise_shape": {"*": None},
                                               "from": [output + '_att_weights'], "dropout": self.attention_dropout}

            d[output + '_att0'] = {"class": "generic_attention", "weights": output + '_att_weights_drop',
                                   "base": 'base:' + output + '_att_value'}  # (B, H, V)
            d[output + '_att_att'] = {"class": "merge_dims", "axes": "static",
                                   "from": [output + '_att0']}  # (B, H*V) except_batch
            d[output + '_att_lin'] = {"class": "linear", "activation": None, "with_bias": False, "from": [output + '_att_att'],
                                      "n_out": self.EncValueTotalDim, "forward_weights_init": self.ff_init}
            d[output + '_att_drop'] = {"class": "dropout", "from": [output + '_att_lin'], "dropout": self.postprocess_dropout}
            d[output + '_att_out'] = {"class": "combine", "kind": "add",
                                      "from": [output + '_self_att_out', output + '_att_drop'],
                                      "n_out": self.EncValueTotalDim}
        elif self.only_one_enc_dec_att == 'add':
            d[output + '_att_out'] = {"class": "combine", "from": [output + '_self_att_out', reuse_att + '_att_drop'],
                    "n_out": self.EncValueTotalDim, "kind": "add"}
        elif self.only_one_enc_dec_att == 'concat':
            d[output + '_att_out'] = {"class": "linear", "from": [output + '_self_att_out', reuse_att + '_att_drop'],
                    "n_out": self.EncValueTotalDim, "activation": None, "with_bias": False, "forward_weights_init": self.ff_init}
        #####
        d[output + '_ff_laynorm'] = {"class": "layer_norm", "from": [output + '_att_out']}
        d[output + '_ff_conv1'] = {"class": "linear", "activation": "relu", "with_bias": True,
                                   "from": [output + '_ff_laynorm'],
                                   "n_out": self.FFDim, "forward_weights_init": self.ff_init}
        d[output + '_ff_conv2'] = {"class": "linear", "activation": None, "with_bias": True,
                                   "from": [output + '_ff_conv1'], "dropout": self.act_dropout,
                                   "n_out": self.EncValueTotalDim, "forward_weights_init": self.ff_init}
        d[output + '_ff_drop'] = {"class": "dropout", "from": [output + '_ff_conv2'], "dropout": self.postprocess_dropout}
        d[output + '_ff_out'] = {"class": "combine", "kind": "add", "from": [output + '_att_out', output + '_ff_drop'],
                                 "n_out": self.EncValueTotalDim}
        d[output] = {"class": "copy", "from": [output + '_ff_out']}

    def build(self):
        network = {
            "source_embed_raw": {"class": "linear", "activation": None, "with_bias": False, "n_out": self.EncValueTotalDim, "forward_weights_init": self.ff_init},
            "source_embed_weighted": {"class": "eval", "from": ["source_embed_raw"],
                                      "eval": "source(0) * %f" % self.embed_weight },
            "source_embed_with_pos": {"class": "positional_encoding", "add_to_input": True,
                                      "from": ["source_embed_weighted"]},
            "source_embed": {"class": "dropout", "from": ["source_embed_with_pos"], "dropout": self.embed_dropout},

            # encoder stack is added by separate function
            "encoder": {"class": "layer_norm", "from": ["enc_%02d" % self.encN]},

            "output": {"class": "rec", "from": [], "unit": {
                'output': {'class': 'choice', 'target': 'classes', 'beam_size': 12, 'from': ["output_prob"], "initial_output": 0}, # this is a vocab_id, make this flexible
                "end": {"class": "compare", "from": ["output"], "value": 0},
                'target_embed_raw': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['prev:output'],
                                     "n_out": self.EncValueTotalDim, "forward_weights_init": self.ff_init},

                # there seems to be no <s> in t2t, they seem to use just the zero vector
                "target_embed_weighted": {"class": "eval", "from": ["target_embed_raw"],
                                          "eval": "source(0) * %f" % self.embed_weight },
                "target_embed_with_pos": {"class": "positional_encoding", "add_to_input": True,
                                          "from": ["target_embed_weighted"]},
                "target_embed": {"class": "dropout", "from": ["target_embed_with_pos"], "dropout": self.embed_dropout},


                # decoder stack is added by separate function
                "decoder": {"class": "layer_norm", "from": ["dec_%02d" % self.decN]},

                "att_selector": {"class": "eval", "from": ["dec_01_att_weights", "dec_02_att_weights", "dec_03_att_weights", "dec_04_att_weights",
                                                      "dec_05_att_weights", "dec_06_att_weights"],
                                  "eval": "self.network.get_config().typed_value('select_random_layer_and_head')([source(i) for i in range(6)], self.network.get_config().bool('in_loop', False))",
                                 "auto_convert": False, "enforce_batch_major": True,
                                  "n_out": 1,
                                  "out_type": {"feature_dim_axis": 1 if not in_loop else 1,
                                               "time_dim_axis": 2 if not in_loop else None,
                                               "batch_dim_axis": 0,
                                               "shape": (1, None, None) if not in_loop else (1, None),
                                               },
                                  },

                # HMM Factorization
                "output_prob" : {"class": "hmm_factorization",
                     "attention_weights": "att_selector",
                     "base_encoder_transformed": "base:encoder",
                     "prev_state": "decoder",
                     "prev_outputs": "prev:target_embed_raw",
                     "top_k": 5,
                     "n_out": num_outputs,
                     "debug": True,
                     #"tie_embedding_weights": "target_embed_raw",
                     #"sample_softmax": 2,
                     #"window_size": 10,  # TODO: remove
                     #"window_factor": 5, # TODO: remove
                     #"attention_location": None, #"/work/smt2/makarov/returnn-hmm/demos/dump",
                     "transpose_and_average_att_weights": False,
                     "target": "classes",
                     "loss": "ce",
                     #"sample_method": "learned_unigram",
                     #"loss": "hmm_factorization_sampled_loss",
                     #"loss_opts": { "num_sampled": 2}
                     },

            }, "target": "classes", "max_seq_len": "max_len_from('base:encoder') * 3", "optimize_move_layers_out": not in_loop},  # TODO: set correct

            "decision": {
                "class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes",
                "loss_opts": {
                    # "debug_print": True
                }
            }

        }

        self.add_trafo_enc_layer(network, "source_embed", "enc_01")
        for n in range(1, self.encN):
            self.add_trafo_enc_layer(network, "enc_%02d" % n, "enc_%02d" % (n+1))

        self.add_trafo_dec_layer(network, network["output"]["unit"], "target_embed", "dec_01")
        for n in range(1, self.decN):
            self.add_trafo_dec_layer(network, network["output"]["unit"], "dec_%02d" % n, "dec_%02d" % (n+1),
                                     reuse_att='dec_01' if self.only_one_enc_dec_att else None)


        return network

network = TransformerNetwork(model_size=128, normalized_loss = True).build()
search_output_layer = "decision"
optimize_move_layers_out = True

debug_print_layer_output_template=True

# trainer
batching = "random"
batch_size = 1000
max_seqs = 5 # TODO: 100
chunking = "0"
truncation = -1
#gradient_clip = 10
#gradient_nan_inf_filter = True
adam = True
gradient_noise = 0.3
learning_rate = 0.0005
learning_rate_control = "newbob"
learning_rate_control_relative_error_relative_lr = True
#model = "./test/transformer-hmm"
model = "/tmp/%s/crnn/%s/model" % (get_login_username(), demo_name)  # https://github.com/tensorflow/tensorflow/issues/6537

num_epochs = 100
save_interval = 1

debug_mode = True
debug_add_check_numerics_ops = True

# log
log_verbosity = 5