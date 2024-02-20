import kappaconfig as kc
from kappaconfig.entities.wrappers import KCScalar


class MinModelPreProcessor(kc.Processor):
    def preorder_process(self, node, trace):
        if len(trace) == 0:
            return
        parent, parent_accessor = trace[-1]
        if isinstance(parent_accessor, str):
            if "model_key" in parent_accessor:
                # replace the value before the first _ with "debug"
                # e.g. model_key: small --> model_key: debug
                # e.g. model_key: small_uneven --> model_key: debug_uneven
                actual = parent[parent_accessor].value
                postfixes = actual.split("_")[1:]
                new_key = "_".join(["debug"] + postfixes)
                parent[parent_accessor] = kc.from_primitive(new_key)
            # "${select:<KEY>:${yaml:models/vit}} -> "${select:debug:${yaml:models/vit}}"
            if isinstance(node, KCScalar) and isinstance(node.value, str):
                if "${select:" in node.value and ":${yaml:models/" in node.value:
                    split = node.value.split(":")
                    if len(split) == 4:
                        node.value = f"{split[0]}:debug:{split[2]}:{split[3]}"
                    elif len(split) == 6:
                        node.value = f"{split[0]}:{split[1]}:{split[2]}:debug:{split[4]}:{split[5]}"
                    else:
                        raise NotImplementedError
