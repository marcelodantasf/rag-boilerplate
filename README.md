# RAG boilerplate

This repository is a learning-oriented boilerplate for independently deployed
RAG services:

- **RAG Core API** owns ingestion, chunking, retrieval, and vector-store access.
- **Embedding API** owns text-to-vector model execution.
- **Qdrant** is private vector-database infrastructure owned by RAG Core.

The architecture references live in [docs/architecture](docs/architecture).

## Container scaffold

\`compose.yaml\` currently starts only Qdrant by default:

\`\`\`sh
docker compose up qdrant
\`\`\`

The two application containers are deliberately in the opt-in \`application\`
profile. They have Dockerfile scaffolding, but no dependency manifest,
application command, or health endpoint yet, so they must not be started until
the services are implemented.

\`\`\`sh
# Use only after both services have an implementation and dependencies.
docker compose --profile application up --build
\`\`\`

Copy \`.env.example\` to \`.env\` for local configuration. Do not commit \`.env\`;
the root \`.gitignore\` excludes it, virtual environments, model/runtime state,
and macOS \`.DS_Store\` files.
