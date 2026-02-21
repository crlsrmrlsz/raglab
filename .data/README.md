# Database State Directory

This directory contains Docker volume mounts for transient infrastructure.

## Structure

```
.data/
├── weaviate/     # Vector database indexes and metadata
└── neo4j/
    ├── data/     # Graph database storage
    ├── logs/     # Neo4j server logs
    └── plugins/  # GDS and APOC plugins (auto-installed)
```

## Important Notes

- These are **generated artifacts** and should not be committed to version control
- This directory is already in `.gitignore`

## Permissions (Linux)

Neo4j containers run as UID 7474. If you encounter permission errors:

```bash
sudo chown -R 7474:7474 .data/neo4j/
```

## Cleanup

To reset all database state:

```bash
docker compose down
rm -rf .data/
docker compose up -d
```

## Migration from Legacy Paths

If you have existing data in the old locations (`weaviate_data/`, `neo4j_data/`, etc.):

```bash
# Stop containers
docker compose down

# Move data to new structure
mkdir -p .data/neo4j
mv weaviate_data .data/weaviate
mv neo4j_data .data/neo4j/data
mv neo4j_logs .data/neo4j/logs
mv neo4j_plugins .data/neo4j/plugins

# Restart
docker compose up -d
```
