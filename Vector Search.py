# Databricks notebook source
# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()
# MAGIC
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC client = VectorSearchClient()
# MAGIC

# COMMAND ----------

# use pre-created index and endpoint

index=client.get_index(endpoint_name='hackathon', index_name='workspace.default.man_index')
endpoint=client.get_endpoint(name='hackathon')

# COMMAND ----------

# Delta Sync Index with embeddings computed by Databricks
results = index.similarity_search(
    query_text="How do I create a connection to other machine",
    columns=["Text", "Summary"],
    num_results=2
    )

results   
