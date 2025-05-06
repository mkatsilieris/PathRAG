import asyncio
import html
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass
from typing import Any, Union, cast
import networkx as nx
import numpy as np
from nano_vectordb import NanoVectorDB
import json
import requests
from urllib.parse import urljoin

from .utils import (
    logger,
    load_json,
    write_json,
    compute_mdhash_id,
)

from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
)


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)
        return left_data

    async def drop(self):
        self._data = {}


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        async def wrapped_task(batch):
            result = await self.embedding_func(batch)
            pbar.update(1)
            return result

        embedding_tasks = [wrapped_task(batch) for batch in batches]
        pbar = tqdm_async(
            total=len(embedding_tasks), desc="Generating embeddings", unit="batch"
        )
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        if len(embeddings) == len(list_data):
            for i, d in enumerate(list_data):
                d["__vector__"] = embeddings[i]
            results = self._client.upsert(datas=list_data)
            return results
        else:
            # sometimes the embedding is not returned correctly. just log it.
            logger.error(
                f"embedding is not 1-1 with data, {len(embeddings)} != {len(list_data)}"
            )

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    @property
    def client_storage(self):
        return getattr(self._client, "_NanoVectorDB__storage")

    async def delete_entity(self, entity_name: str):
        try:
            entity_id = [compute_mdhash_id(entity_name, prefix="ent-")]

            if self._client.get(entity_id):
                self._client.delete(entity_id)
                logger.info(f"Entity {entity_name} have been deleted.")
            else:
                logger.info(f"No entity found with name {entity_name}.")
        except Exception as e:
            logger.error(f"Error while deleting entity {entity_name}: {e}")

    async def delete_relation(self, entity_name: str):
        try:
            relations = [
                dp
                for dp in self.client_storage["data"]
                if dp["src_id"] == entity_name or dp["tgt_id"] == entity_name
            ]
            ids_to_delete = [relation["__id__"] for relation in relations]

            if ids_to_delete:
                self._client.delete(ids_to_delete)
                logger.info(
                    f"All relations related to entity {entity_name} have been deleted."
                )
            else:
                logger.info(f"No relations found for entity {entity_name}.")
        except Exception as e:
            logger.error(
                f"Error while deleting relations for entity {entity_name}: {e}"
            )

    async def index_done_callback(self):
        self._client.save()


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.DiGraph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None
    # def load_nx_graph(file_name) -> nx.Graph:
    #     if os.path.exists(file_name):
    #         return nx.read_graphml(file_name)
    #     return None

    @staticmethod
    def write_nx_graph(graph: nx.DiGraph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {
            node: html.unescape(node.upper().strip()) for node in graph.nodes()
        }  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.DiGraph()
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None
    async def get_node_in_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.in_edges(source_node_id))
        return None
    async def get_node_out_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.out_edges(source_node_id))
        return None
    
    async def get_pagerank(self,source_node_id:str):
        pagerank_list=nx.pagerank(self._graph)
        if source_node_id in pagerank_list:
            return pagerank_list[source_node_id]
        else:
            print("pagerank failed")

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def delete_node(self, node_id: str):
        """
        Delete a node from the graph based on the specified node_id.

        :param node_id: The node_id to delete
        """
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)
            logger.info(f"Node {node_id} deleted from the graph.")
        else:
            logger.warning(f"Node {node_id} not found in the graph for deletion.")

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    # @TODO: NOT USED
    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids
    
    async def edges(self):
        return self._graph.edges()
    async def nodes(self):
        return self._graph.nodes()


@dataclass
class AzureSearchVectorStorage(BaseVectorStorage):
    """
    Vector storage implementation for Azure AI Search.
    Requires:
    - An Azure AI Search service
    - The azure-search-documents package (pip install azure-search-documents)
    """
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        try:
            from azure.search.documents import SearchClient
            from azure.search.documents.models import Vector
            from azure.core.credentials import AzureKeyCredential
        except ImportError:
            raise ImportError("Please install azure-search-documents package: pip install azure-search-documents")

        # Get configuration from global_config or use defaults
        self.endpoint = self.global_config.get("vector_db_storage_cls_kwargs", {}).get("endpoint")
        self.key = self.global_config.get("vector_db_storage_cls_kwargs", {}).get("key")
        self.index_name_prefix = self.global_config.get("vector_db_storage_cls_kwargs", {}).get("index_name_prefix", "pathrag")
        
        if not self.endpoint or not self.key:
            raise ValueError("Azure Search endpoint and key must be provided in vector_db_storage_cls_kwargs")
        
        # Create index name based on namespace
        self.index_name = f"{self.index_name_prefix}-{self.namespace}"
        
        # Initialize the Azure search client
        self.credential = AzureKeyCredential(self.key)
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential
        )
        
        # Create the index if it doesn't exist
        self._ensure_index_exists()
        
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )
        self._max_batch_size = self.global_config.get("embedding_batch_num", 32)
    
    def _ensure_index_exists(self):
        """Ensure the search index exists, create it if it doesn't"""
        try:
            from azure.search.documents.indexes import SearchIndexClient
            from azure.search.documents.indexes.models import (
                SearchIndex,
                SearchField,
                SearchFieldDataType,
                VectorSearch,
                HnswVectorSearchAlgorithmConfiguration,
                VectorSearchProfile,
                VectorSearchAlgorithmKind,
                VectorSearchAlgorithmMetric
            )
        except ImportError:
            raise ImportError("Please install azure-search-documents package: pip install azure-search-documents")
            
        index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=self.credential
        )
        
        try:
            # Check if index exists
            index_client.get_index(self.index_name)
            logger.info(f"Index {self.index_name} already exists")
        except Exception:
            # Create the index
            logger.info(f"Creating index {self.index_name}")
            
            # Define vector search configuration
            vector_search = VectorSearch(
                algorithms=[
                    HnswVectorSearchAlgorithmConfiguration(
                        name="vector-config",
                        kind=VectorSearchAlgorithmKind.HNSW,
                        parameters={
                            "m": 16,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": VectorSearchAlgorithmMetric.COSINE
                        }
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="vector-profile",
                        algorithm_configuration_name="vector-config",
                    )
                ]
            )
            
            # Define fields for the index
            fields = [
                SearchField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
                SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
                SearchField(name="vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                          vector_search_dimensions=self.embedding_func.embedding_dim,
                          vector_search_profile_name="vector-profile"),
            ]
            
            # Add meta fields
            for meta_field in self.meta_fields:
                fields.append(SearchField(
                    name=meta_field, 
                    type=SearchFieldDataType.String, 
                    filterable=True,
                    searchable=True
                ))
                
            # Create the index
            index = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)
            index_client.create_or_update_index(index)
            logger.info(f"Created index {self.index_name}")

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
            
        # Prepare documents for Azure Search
        list_data = []
        for k, v in data.items():
            doc = {"id": k, "content": v.get("content", "")}
            # Add metadata fields
            for meta_field in self.meta_fields:
                if meta_field in v:
                    doc[meta_field] = v[meta_field]
            list_data.append(doc)
            
        # Extract contents for embedding generation
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        # Generate embeddings
        async def wrapped_task(batch):
            result = await self.embedding_func(batch)
            pbar.update(1)
            return result
            
        embedding_tasks = [wrapped_task(batch) for batch in batches]
        pbar = tqdm_async(
            total=len(embedding_tasks), desc="Generating embeddings", unit="batch"
        )
        embeddings_list = await asyncio.gather(*embedding_tasks)
        embeddings = np.concatenate(embeddings_list)
        
        if len(embeddings) == len(list_data):
            # Attach embeddings to documents
            for i, doc in enumerate(list_data):
                doc["vector"] = embeddings[i].tolist()
                
            # Upload documents in batches
            from azure.search.documents.models import IndexDocumentsBatch
            
            batch_size = 1000  # Azure Search batch limit
            results = []
            
            for i in range(0, len(list_data), batch_size):
                batch = list_data[i:i+batch_size]
                upload_result = self.search_client.upload_documents(documents=batch)
                results.extend([r.succeeded for r in upload_result])
                
            logger.info(f"Uploaded {sum(results)} documents to Azure Search")
            return results
        else:
            # Sometimes the embedding is not returned correctly, just log it
            logger.error(
                f"embedding is not 1-1 with data, {len(embeddings)} != {len(list_data)}"
            )
            return []

    async def query(self, query: str, top_k=5):
        # Generate embedding for query
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        
        # Perform vector search
        results = self.search_client.search(
            search_text=None,  # Vector search doesn't need text search
            vector={"vector": embedding.tolist(), "fields": "vector", "k": top_k},
            select=["id", "content"] + list(self.meta_fields),
            top=top_k
        )
        
        # Format results similar to the NanoVectorDB format
        formatted_results = []
        for doc in results:
            result = {
                "id": doc["id"], 
                "content": doc.get("content", ""),
                "distance": doc.get("@search.score", 0)
            }
            # Add metadata fields
            for meta_field in self.meta_fields:
                if meta_field in doc:
                    result[meta_field] = doc[meta_field]
            formatted_results.append(result)
            
        return formatted_results

    async def delete_entity(self, entity_name: str):
        try:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            try:
                self.search_client.delete_documents(documents=[{"id": entity_id}])
                logger.info(f"Entity {entity_name} has been deleted.")
            except Exception:
                logger.info(f"No entity found with name {entity_name}.")
        except Exception as e:
            logger.error(f"Error while deleting entity {entity_name}: {e}")

    async def delete_relation(self, entity_name: str):
        try:
            # Need to find all relations that include this entity
            # Since Azure Search doesn't support complex queries out of the box,
            # we need to retrieve potential relations and filter
            # This assumes src_id and tgt_id are in meta_fields
            results = list(self.search_client.search(
                search_text=entity_name,
                filter=f"src_id eq '{entity_name}' or tgt_id eq '{entity_name}'", 
                select=["id"]
            ))
            
            if results:
                # Delete the found documents
                docs_to_delete = [{"id": doc["id"]} for doc in results]
                self.search_client.delete_documents(documents=docs_to_delete)
                logger.info(f"All relations related to entity {entity_name} have been deleted.")
            else:
                logger.info(f"No relations found for entity {entity_name}.")
        except Exception as e:
            logger.error(f"Error while deleting relations for entity {entity_name}: {e}")

    async def index_done_callback(self):
        # Azure Search indexes are persistent, so no need to save explicitly
        pass
