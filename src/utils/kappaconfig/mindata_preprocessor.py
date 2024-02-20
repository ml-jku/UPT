import kappaconfig as kc
from kappaconfig.entities.wrappers import KCScalar


class MinDataPreProcessor(kc.Processor):
    def preorder_process(self, node, trace):
        if len(trace) == 0:
            return
        parent, parent_accessor = trace[-1]
        if isinstance(parent_accessor, str):
            # datasets (reduce initial loading to a minimum for fast startup)
            if parent_accessor == "datasets":
                pass
