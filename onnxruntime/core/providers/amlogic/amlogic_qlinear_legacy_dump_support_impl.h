Status BuildLegacyQLinearOpsJson(const GraphViewer& graph_viewer, ordered_json& ops_json) {
  ops_json = ordered_json::array();

  std::unordered_map<std::string, TensorState> states;
  std::unordered_map<std::string, RequantRequest> requant_requests;
  std::unordered_map<std::string, std::vector<PendingPadRequest>> pad_requests;
  std::unordered_map<int, TensorState> precomputed_pad_outputs;
  std::unordered_map<std::string, std::vector<PendingPreReshapeTransposeRequest>> pre_reshape_transpose_requests;
  std::unordered_map<int, TensorState> precomputed_reshape_inputs;
  std::unordered_map<std::string, std::vector<PendingPreTransposeRequest>> pre_transpose_requests;
  std::unordered_map<int, TensorState> precomputed_transpose_inputs;
  ORT_RETURN_IF_ERROR(BuildRequantRequests(graph_viewer, requant_requests));
  ORT_RETURN_IF_ERROR(BuildPendingPadRequests(graph_viewer, pad_requests));
  ORT_RETURN_IF_ERROR(BuildPendingPreReshapeTransposeRequests(graph_viewer, pre_reshape_transpose_requests));
  ORT_RETURN_IF_ERROR(BuildPendingPreTransposeRequests(graph_viewer, pre_transpose_requests));

  int conv_id = 0;
  int transpose_id = 0;
  int negative_transpose_id = 1;
  int maxpool_pad_id = 0;
  const int max_node_index = graph_viewer.MaxNodeIndex();
  for (int node_index = 0; node_index < max_node_index; ++node_index) {
    const Node* node = graph_viewer.GetNode(static_cast<NodeIndex>(node_index));
    if (node == nullptr) {
      continue;
    }
    if (IsOrtInsertedQdqHelperNode(*node)) {
      continue;
    }

    bool emitted = false;
    int consumed_extra_nodes = 0;
    ORT_RETURN_IF_ERROR(EmitLegacyNode(graph_viewer, *node, node_index, max_node_index, conv_id, transpose_id,
                                       negative_transpose_id, maxpool_pad_id, states, requant_requests,
                                       pad_requests, precomputed_pad_outputs, pre_reshape_transpose_requests,
                                       precomputed_reshape_inputs, pre_transpose_requests,
                                       precomputed_transpose_inputs, ops_json, emitted, consumed_extra_nodes));
    if (!emitted) {
      continue;
    }
    node_index += consumed_extra_nodes;
  }

  return Status::OK();
}
