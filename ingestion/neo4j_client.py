"""
Thin Neo4j connection wrapper for the actor-interaction graph (usecase.md).

Local dev: a Neo4j Desktop instance (neo4j://127.0.0.1:7687). Prod: the
`neo4j` service in docker/docker-compose.yml, same driver/config shape --
only the URI changes. See configs/config.yaml's `neo4j` block and
.env.example for both.

This module only wraps the connection -- it does not itself decide what the
graph model looks like or migrate anything from Postgres. That's separate,
deliberately: setting up reachable infrastructure and designing/populating
the graph are different steps, and this repo's own history (the untracked
June-2026 `graph.*` Postgres prototype) is a case study in why "there's a
database with data in it" and "there's a reproducible, documented way to
get that data" are not the same claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from neo4j import Driver, GraphDatabase


@dataclass
class Neo4jConfig:
    uri: str = "neo4j://127.0.0.1:7687"
    user: str = "neo4j"
    password: str = "geopulsepass"
    database: str = "neo4j"

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/config.yaml") -> "Neo4jConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        n4j = cfg.get("neo4j", {})
        return cls(
            uri=n4j.get("uri", cls.uri),
            user=n4j.get("user", cls.user),
            password=n4j.get("password", cls.password),
            database=n4j.get("database", cls.database),
        )


class Neo4jClient:
    """Usage:
        with Neo4jClient(Neo4jConfig.from_yaml()) as client:
            client.run("RETURN 1")
    """

    def __init__(self, config: Optional[Neo4jConfig] = None):
        self._config = config or Neo4jConfig()
        self._driver: Optional[Driver] = None

    def connect(self) -> None:
        self._driver = GraphDatabase.driver(
            self._config.uri, auth=(self._config.user, self._config.password)
        )
        self._driver.verify_connectivity()

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def run(self, query: str, parameters: Optional[dict[str, Any]] = None) -> list[dict]:
        if self._driver is None:
            raise RuntimeError("Neo4jClient.connect() must be called before run()")
        with self._driver.session(database=self._config.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def __enter__(self) -> "Neo4jClient":
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()
