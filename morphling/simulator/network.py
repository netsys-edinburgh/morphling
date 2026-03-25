from math import ceil
from .events import NetworkLink, NetworkConfig

class NetworkTopology:
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.links = {(link.source_site, link.target_site): link for link in config.links}

    def get_link(self, src: int, dst: int) -> NetworkLink:
        return self.links.get((src, dst))

    def transfer_time_ns(self, src: int, dst: int, data_bytes: int) -> int:
        if src == dst:
            return 0
        link = self.get_link(src, dst)
        if not link:
            raise ValueError(f"No link from site {src} to {dst}")
        return link.latency_ns + ceil(data_bytes * 8 / (link.bandwidth_gbps * 1e9) * 1e9)

    def reroute_network_cost_ns(self, src: int, dst: int, microbatch_activation_bytes: int) -> int:
        return self.transfer_time_ns(src, dst, microbatch_activation_bytes)

    def reshard_copy_time_ns(self, src: int, dst: int, shard_bytes: int) -> int:
        return self.transfer_time_ns(src, dst, shard_bytes)

def build_default_mesh(num_sites: int, bandwidth_gbps: float = 10.0, latency_ns: int = 500_000) -> NetworkConfig:
    links = []
    for src in range(num_sites):
        for dst in range(num_sites):
            if src != dst:
                links.append(NetworkLink(src, dst, bandwidth_gbps, latency_ns))
    return NetworkConfig(links)

def build_heterogeneous_mesh(site_configs: list[dict]) -> NetworkConfig:
    links = [NetworkLink(c['src'], c['dst'], c['bw'], c['lat']) for c in site_configs]
    return NetworkConfig(links)
