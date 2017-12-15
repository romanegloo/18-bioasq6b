BioASQ6b README
---


### Requirements

#### MetaMap
Locally installed MetaMap is used to obtain MeSH terms. The UMLS concepts 
come with corresponding MeSH tree codes, which are used to get the MeSH codes
 and names. Hence the mapping between the tree codes and MeSH terms is 
 necessary, which can be built by running `/scripts/documents/build_mesh_db.py`.
 This script reads descriptors data file, and creates a mapping table in 
 a SQLite database.
