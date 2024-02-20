import kappaconfig as kc


class MinModelPostProcessor(kc.Processor):
    def preorder_process(self, node, trace):
        if len(trace) == 0:
            return
        if isinstance(node, dict):
            if "initializers" in node:
                i = 0
                while i < len(node["initializers"]):
                    if node["initializers"][i]["kind"] == "pretrained_initializer":
                        del node["initializers"][i]
                    else:
                        i += 1
                if len(node["initializers"]) == 0:
                    node.pop("initializers")
            elif node.get("kind", None) == "offline_fid_callback":
                node["model"] = "dummy"