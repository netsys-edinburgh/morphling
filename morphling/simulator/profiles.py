import json
from .events import NodeProfile, ProfilingData

class ProfileLoader:
    def load_from_json(self, path: str) -> ProfilingData:
        with open(path, 'r') as f:
            data = json.load(f)
        return self.load_from_dict(data)

    def load_from_dict(self, data: dict) -> ProfilingData:
        nodes = [NodeProfile(**n) for n in data['nodes']]
        return ProfilingData(data['model_name'], nodes, data['shard_bytes'])

    def get_node_profile(self, site_id: int, profiles: ProfilingData) -> NodeProfile:
        for node in profiles.nodes:
            if node.site_id == site_id:
                return node
        raise ValueError(f"No profile for site {site_id}")

    def get_step_compute_ns(self, site_id: int, profiles: ProfilingData) -> int:
        return self.get_node_profile(site_id, profiles).step_compute_ns

    def get_activation_bytes(self, site_id: int, profiles: ProfilingData) -> int:
        return self.get_node_profile(site_id, profiles).activation_bytes

    def get_shard_bytes(self, profiles: ProfilingData) -> int:
        return profiles.shard_bytes

def build_homogeneous_profile(model_name: str, num_sites: int, step_compute_ns: int, compute_sms: int, activation_bytes: int, shard_bytes: int) -> ProfilingData:
    nodes = []
    for i in range(num_sites):
        nodes.append(NodeProfile(i, "homogeneous", compute_sms, step_compute_ns, activation_bytes))
    return ProfilingData(model_name, nodes, shard_bytes)
