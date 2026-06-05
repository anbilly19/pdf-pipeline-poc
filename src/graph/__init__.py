"""Knowledge graph package.

Public API:
    build_graph(chunks)            -> nx.DiGraph
    save_graph(graph, path)        -> None
    load_graph(path)               -> nx.DiGraph
    expand_chunks(chunks, graph)   -> list[Chunk]
"""
from src.graph.builder import build_graph, save_graph, load_graph
from src.graph.expander import expand_chunks

__all__ = ["build_graph", "save_graph", "load_graph", "expand_chunks"]
