import tensorflow as tf
from TFNetworkLayer import _ConcatInputLayer, get_concat_sources_data_template
from TFUtil import Data
import numpy as np
import json


class TopicFactorization(_ConcatInputLayer):
  """
  Creates the final distribution out of many sub-distributions for each topic.
  Each topic has a seperate vocab (and embedding), the union of which is the full vocab. Each topic also has a weight.
  The formal derivation is described here:
  https://github.com/nikita68/NMT/blob/master/topic-factorization/Topic_Aware_NMT_Summary.pdf
  Currently, there is a TensorFlow bottleneck which prevents from having the desired speed up.
  """

  layer_class = "topic_factorization"

  def __init__(self, prev_state, prev_outputs, topic_embedding_size, n_out, topics_mapping_file=None,
               debug=False, UNK_INDEX=1, num_topics=None, extract_weight_layers=None, **kwargs):
    """
    Creates the final distribution out of many sub-distributions for each topic.
    Each topic has a seperate vocab (and embedding), the union of which is the full vocab. Each topic also has a weight.
    The formal derivation is described here:
    https://github.com/nikita68/NMT/blob/master/topic-factorization/Topic_Aware_NMT_Summary.pdf
    Currently, there is a TensorFlow bottleneck which prevents from having the desired speed up.
    :param LayerBase prev_state: Current (i-1) state layer.
    :param LayerBase prev_outputs: Previous embedded outputs.
    :param int topic_embedding_size: Embedding size of topic vector.
    :param int n_out: Full vocab size.
    :param str topics_mapping_file: Path to mapping file of which index in full vocabulary corresponds to which topic.
    Mapping file has to be a json dictionary, where each entry is a pair 'key: [token, [topic indices]]',
    with the key being the full_vocab index. The topic indices is a list, as a single vocab can be in multiples topics.
    See ./demos for an example. The sizes of the topics in the mapping do not have to be the same, but the performance
    is limited by the longest mapping!
    :param bool debug: Whether to enable debug mode or not.
    :param int UNK_INDEX: The index of the UNK token.
    :param int num_topics: If no topics_mapping_file is provided, then there will be not splitting of the vocabulary
    into the individual topics, and num_topics amount of weights will be randomly initiliazed and trained.
    :param list[LayerBase] extract_weight_layers: Optionally, the weights for the topic factorization can be extracted
    from previously trained softmax layers. The order of the list should be the same order as the corresponding topics.
    """

    # NOT: that for notational purposes topic_embedding_size = intermediate_size
    super(TopicFactorization, self).__init__(**kwargs)

    # Check if we're in loop or not
    self.in_loop = True if len(prev_state.output.shape) == 1 else False

    # Process inputs
    batch_size, time = self._process_input(prev_state, prev_outputs)

    # Process topics mappings
    topic_vocab_size, mapping_mask, mapping_tensor, num_topics = self._process_mapping(topics_mapping_file, num_topics,
                                                                                       n_out, UNK_INDEX, time,
                                                                                       batch_size)

    # Get topic embeddings
    topic_embeddings = self._process_topic_embeddings(num_topics, topic_embedding_size,
                                                      time, batch_size)  # [(I,) B, num_topics, embedding_size]

    # Get inputs
    word_distribution_weight, input_expanded_trans, word_distribution_bias = \
        self._postprocess_inputs(extract_weight_layers, num_topics, topic_embedding_size, topic_vocab_size)

    # Process and get word distribution
    word_distribution = input_expanded_trans + topic_embeddings  # [(I,) B, num_topics, embedding_size]
    word_distribution = tf.reshape(word_distribution, [-1, num_topics,
                                                       topic_embedding_size])  # [(I *)  B, num_topics, embedding_size]
    word_distribution = tf.transpose(word_distribution, perm=[1, 0, 2])  # [num_topics, (I *) B, embedding_size]

    # [num_topics, (I *) B, embedding_size] x [num_topics, embedding_size, topic_vocab_size]
    # ---------> [num_topics, (I *) B, topic_vocab_size]
    word_distribution = tf.matmul(word_distribution, word_distribution_weight)

    if self.in_loop:
      word_distribution = tf.transpose(word_distribution, [1, 0, 2])
    else:
      word_distribution = tf.reshape(word_distribution, [num_topics, time, batch_size, topic_vocab_size])
      word_distribution = tf.transpose(word_distribution, [1, 2, 0, 3])
    # Now word_distribution is [(I,) B, num_topics, topic_vocab_size]

    # bias
    if word_distribution_bias is None:
      word_distribution_bias = tf.get_variable("word_distribution_bias",
                                               shape=[num_topics, topic_vocab_size],
                                               trainable=True,
                                               initializer=tf.glorot_normal_initializer())
    word_distribution = tf.add(word_distribution, word_distribution_bias)

    if topics_mapping_file is not None:
      word_distribution = tf.where(mapping_mask, x=word_distribution, y=tf.fill(tf.shape(word_distribution),
                                                                                tf.float32.min))

    # Apply softmax
    word_distribution = tf.nn.softmax(word_distribution)  # [(I,) B, num_topics, topic_vocab_size]

    # Get topic distribution
    topic_distribution = self.linear(x=self.prev_outputs + self.prev_state,
                                     units=num_topics,
                                     inp_dim=prev_state.output.dim)
    topic_distribution = tf.nn.softmax(topic_distribution)  # [(I,) B, num_topics]

    # Debug
    if debug:
      if topics_mapping_file is not None:
        topic_distribution = tf.Print(topic_distribution, [mapping_mask[0]], message="Mapping mask: ",
                                      summarize=10)
      topic_distribution = tf.Print(topic_distribution, [topic_distribution[0]], message="Topic Distribution:",
                                    summarize=10)
      topic_distribution = tf.Print(topic_distribution, [word_distribution[0]], message="Word Distributions:",
                                    summarize=10)

    # Make topic_distribution correct size
    topic_distribution = tf.expand_dims(topic_distribution, axis=-1)  # [(I,) B, num_topics, 1]

    if topics_mapping_file is None:
      final_distribution = self._postprocess_topic_distribution(topic_distribution, word_distribution, debug)
    else:
      final_distribution = self._postprocess_topic_mapping(topic_distribution, topic_vocab_size, time, batch_size,
                                                           word_distribution, num_topics, mapping_tensor, n_out)

    # final_distribution is now [(I,) B, vocab_size]
    if debug:
      final_distribution = tf.Print(final_distribution, [final_distribution[0]], message="Final distribution: ",
                                    summarize=100)
      final_distribution = tf.Print(final_distribution, [tf.reduce_sum(final_distribution, axis=-1)],
                                    message="Sanity check", summarize=1000)

    # Set output info
    self.output.placeholder = final_distribution

    # Add all trainable params
    with self.var_creation_scope() as scope:
      self._add_all_trainable_params(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name))

  def _process_input(self, prev_state, prev_outputs):
    """
    Processes the input.
    :param LayerBase prev_state: Current (i-1) state layer.
    :param LayerBase prev_outputs: Previous embedded outputs.
    :return: batch_size, time
    """
    # Get data
    if self.in_loop is False:
      self.prev_state = prev_state.output.get_placeholder_as_time_major()  # [I, B, intermediate_size]
      self.prev_outputs = prev_outputs.output.get_placeholder_as_time_major()  # [I, B, intermediate_size]
      shape = tf.shape(self.prev_outputs)
      time = shape[0]
      batch_size = shape[1]
    else:
      self.prev_state = prev_state.output.get_placeholder_as_batch_major()  # [B, intermediate_size]
      self.prev_outputs = prev_outputs.output.get_placeholder_as_batch_major()  # [B, intermediate_size]
      shape = tf.shape(self.prev_outputs)
      batch_size = shape[0]
      time = -1

    return batch_size, time

  def _process_mapping(self, topics_mapping_file, num_topics, n_out, UNK_INDEX, time, batch_size):
    """
    Process the mapping file.
    :param str topics_mapping_file: Path to topic mappings file.
    :param int num_topics: Number of topics, if applicable
    :param int n_out: Full vocab size.
    :param int UNK_INDEX: Index of UNK token.
    :param tf.Tensor[int] time: Time size.
    :param tf.Tensor[int] batch_size: Batch size.
    :return: topic_vocab_size, mapping_mask, mapping_tensor, num_topics
    """
    if topics_mapping_file is None:
      assert num_topics is not None, "TF: if topics_mapping_file not set, then num_topics needs to explicitly set"
      topic_vocab_size = n_out
      mapping_mask = mapping_tensor = None
    else:
      # Get topic info
      mapping_tensor, num_topics, topic_vocab_size, mapping_mask = \
        self.load_topic_mapping_file(mapping_file_path=topics_mapping_file, UNK_INDEX=UNK_INDEX)

      # Modify mask
      if self.in_loop is False:
        mapping_mask = tf.expand_dims(tf.expand_dims(mapping_mask, axis=0), axis=0)
        mapping_mask = tf.tile(mapping_mask, [time, batch_size, 1, 1])
      else:
        mapping_mask = tf.expand_dims(mapping_mask, axis=0)
        mapping_mask = tf.tile(mapping_mask, [batch_size, 1, 1])
    return topic_vocab_size, mapping_mask, mapping_tensor, num_topics

  def _process_topic_embeddings(self, num_topics, topic_embedding_size, time, batch_size):
    """
    Process and retrieve topic embeddings.
    :param int num_topics: Number of topics, if applicable
    :param int topic_embedding_size: Size of topic embeddings.
    :param tf.Tensor[int] time: Time size.
    :param tf.Tensor[int] batch_size: Batch size.
    :return: topic_embeddings [(I,) B, num_topics, embedding_size]
    """

    topic_embeddings = tf.get_variable("word_embeddings", [1, num_topics, topic_embedding_size])
    if self.in_loop is False:
      topic_embeddings = tf.expand_dims(topic_embeddings, axis=0)
      tiling = [time, batch_size, 1, 1]
    else:
      tiling = [batch_size, 1, 1]
    topic_embeddings = tf.tile(topic_embeddings, tiling)  # [(I,) B, num_topics, embedding_size]
    return topic_embeddings

  def _postprocess_inputs(self, extract_weight_layers, num_topics, topic_embedding_size, topic_vocab_size):
    """
    Post process the inputs to the correct shapes.
    :param list[LayerBase] extract_weight_layers: Optionally, the weights for the topic factorization can be extracted
    from previously trained softmax layers. The order of the list should be the same order as the corresponding topics.
    :param int num_topics: Number of topics, if applicable
    :param int topic_embedding_size: Size of topic embeddings.
    :param int topic_vocab_size: Max size of a topic's vocab.
    :return: word_distribution_weight [num_topics, embedding_size, topic_vocab_size], input_expanded_trans,
    word_distribution_bias
    """

    if self.in_loop is False:
      input_expanded = tf.tile(tf.expand_dims(self.prev_outputs + self.prev_state, axis=3), [1, 1, 1, num_topics])
      perm = [0, 1, 3, 2]
    else:
      input_expanded = tf.tile(tf.expand_dims(self.prev_outputs + self.prev_state, axis=2), [1, 1, num_topics])
      perm = [0, 2, 1]

    # Get word distribution per topic
    # Now [(I,) B, embedding_size, num_topics]
    input_expanded_trans = tf.transpose(input_expanded, perm=perm)  # Now [(I,) B, num_topics, embedding_size]
    word_distribution_bias = None

    if extract_weight_layers:
      assert len(extract_weight_layers) == num_topics, "TF: Amount of topics must be equal to the amount of layers" \
                                                       "provided in extract_weight_layers"
      word_distribution_weight, word_distribution_bias = self.restore_and_stack_weight_matrix(
        extract_weight_layers=extract_weight_layers)
      # [num_topics, embedding_size, topic_vocab_size], [num_topics, topic_vocab_size]
    else:
      # https://github.com/tensorflow/tensorflow/issues/17149 can't use tf.layers.dense
      word_distribution_weight = tf.get_variable("word_distribution_weight",
                                                 shape=[num_topics, topic_embedding_size, topic_vocab_size],
                                                 trainable=True,
                                                 initializer=tf.glorot_normal_initializer())

    return word_distribution_weight, input_expanded_trans, word_distribution_bias

  def _postprocess_topic_mapping(self, topic_distribution, topic_vocab_size, time, batch_size, word_distribution,
                                 num_topics, mapping_tensor, n_out):
    """
    Post process the topic mappings into a single distribution.
    :param tf.Tensor topic_distribution: Topic distribution tensor. [(I,) B, num_topics]
    :param int topic_vocab_size: Max size of a topic's vocab.
    :param tf.Tensor[int] time: Time size.
    :param tf.Tensor[int] batch_size: Batch size.
    :param tf.Tensor word_distribution: Distribution over topic vocab [(I,) B, num_topics, topic_vocab_size]
    :param int num_topics: Number of topics, if applicable
    :param ndarray mapping_tensor: Mapping tensor which topic element goes where. [num_topics, topic_vocab_size]
    :param int n_out: Full vocab size.
    :return: tf.Tensor final_distribution [(I,) B, vocab_size]
    """

    if self.in_loop:
      topic_distribution = tf.tile(topic_distribution, [1, 1, topic_vocab_size])
    else:
      topic_distribution = tf.tile(topic_distribution, [1, 1, 1, topic_vocab_size])

    # Slice wise multiplication to normalize the distributions accordingly
    final_distribution = tf.multiply(word_distribution, topic_distribution)  # [(I,) B, num_topics, topic_vocab_size]

    # Optimization using mapping
    if self.in_loop is False:
      # Get indices
      # TODO: optimize this meshgrid somehow? maybe precalculate it
      # TODO: see if there is alternative
      # TODO: meshgrid is performance bottleneck
      ii, bb, _, _ = tf.meshgrid(tf.range(time), tf.range(batch_size),
                                 tf.range(num_topics), tf.range(topic_vocab_size),
                                 indexing='ij')

      # Make mapping of shape [I, B, num_topics, topic_vocab_size]
      mapping_tensor = tf.expand_dims(tf.expand_dims(mapping_tensor, axis=0), axis=0)
      mapping_tensor = tf.tile(mapping_tensor, [time, batch_size, 1, 1])

      # Get final indices
      mapping_tensor = tf.stack([ii, bb, mapping_tensor], axis=-1)

      # Finally retrieve final distribution
      final_distribution = tf.scatter_nd(indices=mapping_tensor, updates=final_distribution,
                                         shape=[time, batch_size, n_out])
    else:
      # Get indices
      bb, _, _ = tf.meshgrid(tf.range(batch_size), tf.range(num_topics), tf.range(topic_vocab_size),
                                 indexing='ij')

      # Make mapping of shape [B, num_topics, topic_vocab_size]
      mapping_tensor = tf.expand_dims(mapping_tensor, axis=0)
      mapping_tensor = tf.tile(mapping_tensor, [batch_size, 1, 1])

      # Get final indices
      mapping_tensor = tf.stack([bb, mapping_tensor], axis=-1)

      # Finally retrieve final distribution
      final_distribution = tf.scatter_nd(indices=mapping_tensor, updates=final_distribution,
                                         shape=[batch_size, n_out])

    return final_distribution

  def _postprocess_topic_distribution(self, topic_distribution, word_distribution, debug):
    """
    Calculate final distribution when no mapping tensor is used, i.e. topic_vocab_size = vocab_size.
    :param tf.Tensor topic_distribution: Topic distribution tensor. [(I,) B, num_topics]
    :param tf.Tensor word_distribution: Distribution over topic vocab [(I,) B, num_topics, topic_vocab_size]
    :param bool debug: Whether to use debug mode.
    :return: tf.Tensor final_distribution [(I,) B, vocab_size]
    """

    # Assume full softmax, so topic_vocab_size = vocab_size
    # word_distribution [(I,) B, num_topics, vocab_size]
    # topic_distribution [(I,) B, num_topics, 1]
    final_distribution = tf.matmul(topic_distribution, word_distribution, transpose_a=True)
    # final_distribution [(I,) B, 1, topic_vocab_size]
    final_distribution = tf.squeeze(final_distribution, axis=-2)  # [(I,) B, vocab_size]

    if debug:
      final_distribution = tf.Print(final_distribution, [tf.shape(topic_distribution),
                                                         tf.shape(word_distribution),
                                                         tf.shape(final_distribution)],
                                    summarize=100, message="Full softmax shapes (topic_dis, word_dis, final_dis): ")
    return final_distribution

  def _add_all_trainable_params(self, tf_vars):
    for var in tf_vars:
      self.add_param(param=var, trainable=True, saveable=True)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    d["from"] = d["prev_state"]
    super(TopicFactorization, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["prev_state"] = get_layer(d["prev_state"])
    d["prev_outputs"] = get_layer(d["prev_outputs"])
    if "extract_weight_layers" in d:
      d["extract_weight_layers"] = [get_layer(s) for s in d["extract_weight_layers"]]

  @classmethod
  def get_out_data_from_opts(cls, prev_state, n_out, out_type=None, sources=(), **kwargs):

    in_loop = True if len(prev_state.output.shape) == 1 else False

    data = prev_state.output

    if in_loop is False:
      data = data.copy_as_time_major()  # type: Data
      data.shape = (None, n_out)
      data.time_dim_axis = 0
      data.batch_dim_axis = 1
      data.dim = n_out
    else:
      data = data.copy_as_batch_major()  # type: Data
      data.shape = (n_out,)
      data.batch_dim_axis = 0
      data.time_dim_axis = None
      data.dim = n_out
    return data

  def load_topic_mapping_file(self, mapping_file_path, UNK_INDEX):
    """
    Loads and shapes the topic mapping file.
    :param str mapping_file_path: Path to mapping file of which index in full vocabulary corresponds to which topic.
    Mapping file has to be a json dictionary, where each entry is a pair 'key: [token, [topic indices]]',
    with the key being the full_vocab index. The topic indices is a list, as a single vocab can be in multiples topics.
    See ./demos for an example. The sizes of the topics in the mapping do not have to be the same, but the performance
    is limited by the longest mapping!
    :param int UNK_INDEX: Index of UNK token in full vocab.
    :return: ndarray mapping_tensor: Mapping tensor which topic element goes where. [num_topics, topic_vocab_size]
    """

    # Mapping file has to be a json dictionary, where each entry is a pair (token, topic index), with the key being the
    # full_vocab index
    # Token is used for debugging purposes only
    # Each entry contains a list of topic indices (starting from 0)
    with open(mapping_file_path, "r") as w:
      loaded_dic = json.load(w)

    full_dic = {}  # Each entry in full_dic is the id of a topic, and returns a list of ids of the word ids in this list

    # Makes loaded_dic sorted and consistent between loads
    loaded_dic_keys = [int(w) for w in list(loaded_dic.keys())]
    loaded_dic_keys.sort()
    loaded_dic_keys = [str(w) for w in loaded_dic_keys]  # So hacky

    for full_vocab_idx in loaded_dic_keys:
      topics = loaded_dic[full_vocab_idx][1]
      for t in topics:
        if t in full_dic:
          full_dic[t].append(full_vocab_idx)
        else:
          full_dic[t] = [full_vocab_idx]

    # Pad with mappings to UNK to make same size
    mask = []
    amount_of_topics = len(full_dic.keys())

    max_len = max([len(full_dic[idx]) for idx in range(amount_of_topics)])
    for topic_idx in range(amount_of_topics):
      curr_len = len(full_dic[topic_idx])
      amount_to_pad = max_len - curr_len
      # Pad UNK
      full_dic[topic_idx].extend([UNK_INDEX] * amount_to_pad)
      # make mask
      mask.append([True] * curr_len)
      mask[-1].extend([False] * amount_to_pad)
    mask = np.asarray(mask)  # Of shape [num_topics, topic_vocab_size]

    # Sizes
    topic_vocab_size = len(full_dic[0])
    test_size = len(full_dic[0])
    for i in range(1, amount_of_topics):
      assert test_size == len(full_dic[i]), "Topic Factorization: Topic not of equal sizes!"

    # Before scatter op size is [(I,) B, num_topics, topic_vocab_size]
    # Final vocab size should be [(I,) B, full_vocab]

    # Index tensor should be [(I,) B, num_topic, topic_vocab_size, (3)2],
    # Last dim of index contains info where the corresponding data goes into final distribution
    # ie. [(I,) B, num_topic, topic_vocab_size, [(I,) B, full_vocab]]

    # Tensor which we make with this function is of size [num_topics, topic_vocab_size]

    full_tensor = []
    for topic_id in range(amount_of_topics):
      topic_words = [np.int32(w) for w in full_dic[topic_id]]
      full_tensor.append(topic_words)
    mapping_tensor = np.asarray(full_tensor)  # [num_topics, topic_vocab_size]

    print("Mapping tensor:")
    print(mapping_tensor)
    print("Shape of mapping tensor (num_topics, topic_vocab_size): " + str(mapping_tensor.shape))

    return mapping_tensor, amount_of_topics, topic_vocab_size, mask

  def linear(self, x, units, inp_dim=None, weight=None, bias=False):
    """
    An optimized GPU linear layer.
    :param tf.Tensor x: Input
    :param int units: Output feature dim.
    :param int inp_dim: If weight not provided, then the input dimension
    :param tf.Tensor weight: If set, then use this weight to multiply, else will generate custom trainable weight.
    :param bool,tf.Tensor bias: Provide a tensor to use as bias or set to true for auto generated trainable bias.
    :return: Output tensor.
    """

    in_shape = tf.shape(x)
    inp = tf.reshape(x, [-1, in_shape[-1]])
    if weight is None:
      assert inp_dim is not None, "TF: Linear layer: if weight is none, inp_dim musn't be None"
      weight = tf.get_variable("lin_weight", trainable=True, shape=[inp_dim, units], dtype=tf.float32)
    out = tf.matmul(inp, weight)
    if bias is True or bias is not None:
      if bias is True:
        bias = tf.get_variable("lin_bias", trainable=True, initializer=tf.zeros_initializer, shape=[units],
                               dtype=tf.float32)
      out = out + bias
    out_shape = tf.concat([in_shape[:-1], [units]], axis=0)
    out = tf.reshape(out, out_shape)
    # TODO: maybe use tensordot instead
    return out

  def restore_and_stack_weight_matrix(self, extract_weight_layers, weight_suffix="W:0",
                                      bias_suffix="b:0"):
    """
    Restores variables from checkpoints and stacks them, to be used as weights for the individual topic distributions.
    :param list[LayerBase] extract_weight_layers: Optionally, the weights for the topic factorization can be extracted
    from previously trained softmax layers. The order of the list should be the same order as the corresponding topics.
    :return: full_tensor [num_topics, embedding_size, topic_vocab_size], bias_tensor [num_topics, topic_vocab_size]
    """

    # create softmax layers in config with custom import which don't have any used output
    # NOTE: all variables have to be trainable! Setting trainable=False in the config has no effect

    # get variables from layers
    tensor_list = []
    for layer in extract_weight_layers:
      w = tf.get_default_graph().get_tensor_by_name(
                          layer.get_base_absolute_name_scope_prefix() + weight_suffix)
      tensor_list.append(w)

    # stack them into appropriate weight matrix
    full_tensor = tf.stack(tensor_list, axis=0)

    # Bias
    all_bias = [l.with_bias for l in extract_weight_layers]
    assert len(set(all_bias)) == 1, "TF: Either all layers need to have a bias, or none!"

    bias_tensor = None

    if all(all_bias):
      bias_list = []
      for layer in extract_weight_layers:
        w = tf.get_default_graph().get_tensor_by_name(
                            layer.get_base_absolute_name_scope_prefix() + bias_suffix)
        bias_list.append(w)

      bias_tensor = tf.stack(bias_list, axis=0)

    return full_tensor, bias_tensor


