"""EmbeddingAgent + ClusteringAgent — multilingual embeddings, embedding-based
near-dup removal, UMAP reduction, HDBSCAN clustering, TF-IDF top terms.
Cluster labels are assigned later by the InsightAgent's LLM pass."""

from __future__ import annotations

import numpy as np
from rich.console import Console
from sklearn.feature_extraction.text import TfidfVectorizer

from ...core import config
from ...core.models import Cluster, ProcessedDocument, RunManifest, utcnow
from ...core.storage import Store, new_run_id

console = Console()


def _embed(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=128, show_progress_bar=True,
                        normalize_embeddings=True)


def _near_dup_pass(store: Store, docs: list[ProcessedDocument], emb: np.ndarray,
                   threshold: float) -> tuple[list[ProcessedDocument], np.ndarray, int]:
    """Drop docs whose embedding is ~identical to an earlier one (copy-paste
    reviews with trivial edits). Order-stable: first occurrence is canonical."""
    kept_idx: list[int] = []
    dropped: list[ProcessedDocument] = []
    # cosine sim on normalized vectors = dot product; chunk to bound memory
    kept_matrix: list[np.ndarray] = []
    for i, d in enumerate(docs):
        v = emb[i]
        is_dup = False
        if kept_matrix:
            block = np.vstack(kept_matrix[-5000:])  # recent window is enough in practice
            sims = block @ v
            j = int(np.argmax(sims))
            if sims[j] >= threshold:
                canon = docs[kept_idx[max(0, len(kept_idx) - 5000) + j]]
                d.kept = False
                d.drop_reason = "near_dup"
                d.dup_of = canon.id
                dropped.append(d)
                is_dup = True
        if not is_dup:
            kept_idx.append(i)
            kept_matrix.append(v)
    store.upsert_processed(dropped)
    return [docs[i] for i in kept_idx], emb[kept_idx], len(dropped)


def run_clustering(store: Store) -> RunManifest:
    cfg = config.settings()["processing"]
    manifest = RunManifest(agent="clustering", run_id=new_run_id("clustering"),
                           started_at=utcnow())

    informative = {e.id for e in store.iter_enriched() if e.is_informative}
    docs = [d for d in store.iter_processed(kept_only=True)
            if not informative or d.id in informative]
    if not informative:
        manifest.notes.append("no enrichment found — clustering ALL kept docs")
    texts = [d.clean_text for d in docs]
    console.print(f"embedding {len(texts)} documents with {cfg['embedding_model']} …")
    emb = _embed(texts, cfg["embedding_model"])

    docs, emb, n_near = _near_dup_pass(store, docs, emb, cfg["near_dup_cosine_threshold"])
    console.print(f"near-dup pass removed {n_near}; {len(docs)} docs to cluster")

    np.save(config.PROCESSED_DIR / "embeddings.npy", emb)
    with open(config.PROCESSED_DIR / "embedding_ids.txt", "w") as f:
        f.write("\n".join(d.id for d in docs))

    import umap
    from sklearn.cluster import HDBSCAN

    reducer = umap.UMAP(
        n_neighbors=cfg["umap"]["n_neighbors"],
        n_components=cfg["umap"]["n_components"],
        min_dist=cfg["umap"]["min_dist"],
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(emb)
    labels = HDBSCAN(
        min_cluster_size=cfg["hdbscan"]["min_cluster_size"],
        min_samples=cfg["hdbscan"]["min_samples"],
    ).fit_predict(reduced)

    clusters: list[Cluster] = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue  # noise stays unclustered, never forced in
        member_ids = [docs[i].id for i in np.where(labels == cid)[0]]
        member_texts = [texts_i for i, texts_i in
                        zip(range(len(docs)), (d.clean_text for d in docs))
                        if labels[i] == cid]
        top_terms: list[str] = []
        try:
            tfidf = TfidfVectorizer(max_features=8, stop_words="english",
                                    ngram_range=(1, 2), min_df=2)
            tfidf.fit(member_texts)
            top_terms = list(tfidf.get_feature_names_out())
        except ValueError:
            pass
        clusters.append(Cluster(cluster_id=int(cid), size=len(member_ids),
                                member_ids=member_ids, top_terms=top_terms))

    store.replace_clusters(clusters)
    n_noise = int((labels == -1).sum())
    manifest.counts = {"embedded": len(docs), "near_dups_removed": n_near,
                       "clusters": len(clusters), "unclustered_noise": n_noise}
    manifest.status = "ok"
    manifest.finished_at = utcnow()
    store.save_manifest(manifest)
    console.print(f"[green]✓ clustering[/green] {len(clusters)} clusters, "
                  f"{n_noise} docs left as noise")
    return manifest
