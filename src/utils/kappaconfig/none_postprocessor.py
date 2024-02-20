import kappaconfig as kc


class NonePostProcessor(kc.Processor):
    def preorder_process(self, node, trace):
        if len(trace) == 0:
            return
        if isinstance(node, str) and node.lower() == "none":
            parent, accessor = trace[-1]
            parent[accessor] = None
