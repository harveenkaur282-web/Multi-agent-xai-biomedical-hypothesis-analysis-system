import os
import json
import re
import numpy as np
import spacy
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer  
from sklearn.metrics.pairwise import cosine_similarity
from pipeline.state import PCOSState
from utils.pubmed_client import search_pubmed_pcos
from utils.neo4j_client import Neo4jMedicalGraph 

try:
    nlp = spacy.load("en_ner_bc5cdr_md")
except OSError:
    import spacy.cli
    spacy.cli.download("en_ner_bc5cdr_md")
    nlp = spacy.load("en_ner_bc5cdr_md")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def tokenize(text: str) -> list:
    return re.findall(r'\b[a-z0-9]+\b', text.lower())

def extract_biomedical_entities(text: str) -> list:
    """Uses SciSpacy clinical NER to isolate verified active medical symptoms with bidirectional negation checking."""
    doc = nlp(text)
    discovered = []
    text_lower = text.lower()
    
    for ent in doc.ents:
        ent_lower = ent.text.lower()
        
        ent_start = text_lower.find(ent_lower)
        if ent_start == -1:
            continue
        ent_end = ent_start + len(ent_lower)
        
        # BIDIRECTIONAL NEGATION & OPTIMALITY GUARD WINDOW
        look_back = text_lower[max(0, ent_start - 30):ent_start]
        look_ahead = text_lower[ent_end:min(len(text_lower), ent_end + 40)]
        full_context = look_back + " [ENTITY] " + look_ahead
        
        stop_words = ["no ", "normal", "optimal", "excellent", "negative", "clear", "exceptional", "stable"]
        if any(word in full_context for word in stop_words):
            continue
            
        if ent_lower in ["treatment-resistant", "bilateral", "chronic", "prominent"]:
            continue
            
        # FIX: Feed both variants to make sure Neo4j property matches succeed regardless of index configuration
        discovered.append(ent.text.title())
        discovered.append(ent.text.lower())
        
    return list(set(discovered))

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    chunks = []
    if not text:
        return chunks
    step = max(1, chunk_size - overlap)
    for i in range(0, len(text), step):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
        if i + chunk_size >= len(text):
            break
    return chunks


def node1_ingestion_fn(state: PCOSState) -> dict:
    raw_case = dict(state["raw_input"])
    
    if "labs" in raw_case and isinstance(raw_case["labs"], dict):
        labs = raw_case["labs"]
        mapping = {
            "fasting_insulin_uiu_ml": ["fasting_insulin"],
            "lh_fsh_ratio": ["lh_fsh_ratio"],
            "testosterone_ng_dl": ["free_testosterone", "testosterone"],
            "free_testosterone": ["free_testosterone", "testosterone"],
            "testosterone": ["free_testosterone", "testosterone"],
            "amh_ng_ml": ["amh_levels"],
            "amh_levels": ["amh_levels"],
            "homa_ir": ["homa_ir"],
        }
        for nested_key, flat_keys in mapping.items():
            if nested_key in labs:
                for fk in flat_keys:
                    if fk not in raw_case:
                        raw_case[fk] = labs[nested_key]
        if "bmi" in labs and "bmi" not in raw_case:
            raw_case["bmi"] = labs["bmi"]
            
    state["raw_input"] = raw_case

    remarks = raw_case.get("clinical_remarks", "")
    verified_entities = extract_biomedical_entities(remarks)
    print(f" [DIAGNOSTIC] SciSpacy Extracted Entities: {verified_entities}")
    kg_substructure = []
    try:
        db = Neo4jMedicalGraph()
        kg_substructure = db.get_clinical_subgraph(verified_entities)
        print(f"[DIAGNOSTIC] Neo4j Subgraph Raw Return: {kg_substructure}")
        print(f"[RAG Ingestion Node] Successfully extracted {len(kg_substructure)} graph cached medical pathways.")
        db.close()
    except Exception as e:
        print(f"[Ingestion Node Error] Failed to initialize or query Neo4j: {e}") 
    
    if not isinstance(kg_substructure, list):
        kg_substructure = [kg_substructure] if kg_substructure else []

    base_boolean_terms = '("PCOS" OR "polycystic ovary syndrome")'
    symptom_boolean_tokens = []
    semantic_vector_tokens = ["PCOS", "polycystic ovary syndrome"]
    
    junk_words = ["bilateral", "treatment-resistant", "prominent", "severe", "clinical"]
    
    for ent in verified_entities:
        clean_ent = ent.lower().replace("polycystic ovary syndrome", "").strip()
        clean_ent = clean_ent.replace("pcos", "").strip()
        
        for word in junk_words:
            clean_ent = clean_ent.replace(word, "").strip()
            
        clean_ent = " ".join(clean_ent.split())
        
        if clean_ent and len(clean_ent) > 2:
            symptom_boolean_tokens.append(f'"{clean_ent}"')
            semantic_vector_tokens.append(clean_ent)
            
    config_path = os.path.join("data", "pcos_thresholds.json")
    try:
        with open(config_path) as f:
            config_data = json.load(f)
            thresholds = config_data["lab_thresholds"]
            insulin_threshold = float(thresholds["fasting_insulin_uiu_ml"]["elevated_min"])
            lh_fsh_threshold = float(thresholds["lh_fsh_ratio"]["elevated_min"])
    except Exception:
        insulin_threshold = 14.0
        lh_fsh_threshold = 2.0

    if float(raw_case.get("fasting_insulin", 0.0)) >= insulin_threshold:
        symptom_boolean_tokens.append('"hyperinsulinemia"')
        semantic_vector_tokens.append("hyperinsulinemia")
    if float(raw_case.get("lh_fsh_ratio", 0.0)) >= lh_fsh_threshold:
        symptom_boolean_tokens.append('"lh-fsh ratio"')
        semantic_vector_tokens.append("lh-fsh ratio")

    semantic_vector_tokens = list(set(semantic_vector_tokens))
    semantic_vector_query = " ".join(semantic_vector_tokens).strip()
    
    if symptom_boolean_tokens:
        symptom_boolean_tokens = list(set(symptom_boolean_tokens))
        api_search_query = f"{base_boolean_terms} AND ({' OR '.join(symptom_boolean_tokens)})"
    else:
        api_search_query = base_boolean_terms
        
    print(f"[RAG Ingestion] Production API Target: '{api_search_query}'")
    print(f"[RAG Ingestion] Production Vector Target: '{semantic_vector_query}'")
    
    use_live_search = raw_case.get("use_live_search", False)
    documents = []
    if use_live_search:
        try:
            documents = search_pubmed_pcos(api_search_query)
        except Exception as api_err:
            print(f"[RAG Ingestion] PubMed live API failed: {api_err}. Loading local cache fallback.")
        if not documents:
            cache_path = os.path.join("data", "pubmed_cache.json")
            try:
                with open(cache_path, "r") as cache_file:
                    documents = json.load(cache_file)
            except Exception as cache_err:
                print(f"[RAG Ingestion] Local cache also unavailable: {cache_err}")
    else:
        corpus_path = os.path.join("data", "pcos_literature_corpus.json")
        try:
            with open(corpus_path, "r", encoding="utf-8") as corpus_file:
                documents = json.load(corpus_file)
        except Exception as corpus_err:
            print(f"[RAG Ingestion] Local corpus unavailable: {corpus_err}")

    unified_docs = []
    for doc in documents:
        text = doc.get("text") or doc.get("abstract") or ""
        title = doc.get("title") or "Untitled"
        pmid = doc.get("pmid") or ""
        if text:
            unified_docs.append({
                "title": title,
                "text": text,
                "pmid": pmid
            })

    if not unified_docs:
        print("[RAG Ingestion Node] Successfully appended 0 ranked papers to state payload.")
        return {
            "retrieved_chunks": [],
            "graph_knowledge": kg_substructure
        }

    all_chunks = []
    for doc_idx, doc in enumerate(unified_docs, start=1):
        chunks = chunk_text(doc["text"])
        for chunk_idx, chunk_text_content in enumerate(chunks, start=1):
            all_chunks.append({
                "id": f"[Source-{doc_idx}, Chunk-{chunk_idx}]",
                "text": chunk_text_content,
                "title": doc["title"],
                "pmid": doc["pmid"]
            })

    chunk_texts = [c["text"] for c in all_chunks]
    query_embedding = embedding_model.encode([semantic_vector_query])
    chunk_embeddings = embedding_model.encode(chunk_texts)
    dense_scores = cosine_similarity(query_embedding, chunk_embeddings).flatten()
    dense_indices = np.argsort(dense_scores)[::-1]
    dense_ranks = {idx: rank for rank, idx in enumerate(dense_indices, start=1)}

    tokenized_chunks = [tokenize(c["text"]) for c in all_chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    bm25_scores = bm25.get_scores(tokenize(semantic_vector_query))
    bm25_indices = np.argsort(bm25_scores)[::-1]
    bm25_ranks = {idx: rank for rank, idx in enumerate(bm25_indices, start=1)}

    rrf_scores = []
    for idx in range(len(all_chunks)):
        rank_bm25 = bm25_ranks[idx]
        rank_dense = dense_ranks[idx]
        rrf_score = 1.0 / (60.0 + rank_bm25) + 1.0 / (60.0 + rank_dense)
        rrf_scores.append(rrf_score)

    max_theoretical = 2.0 / 61.0
    merged_results = []
    seen_texts = set()
    seen_titles = set()
    sorted_indices = np.argsort(rrf_scores)[::-1]
    for idx in sorted_indices:
        chunk = all_chunks[idx]
        normalized_text = " ".join(chunk["text"].strip().lower().split())
        normalized_title = chunk["title"].strip().lower()
        
        # Skip if we've seen this exact text OR another chunk from the same paper
        if normalized_text in seen_texts or normalized_title in seen_titles:
            continue
        seen_texts.add(normalized_text)
        seen_titles.add(normalized_title)
        
        normalized_rrf = rrf_scores[idx] / max_theoretical
        presentation_score = round(0.3 + (normalized_rrf * 0.7), 4)
        merged_results.append({
            "id": chunk["id"],
            "title": chunk["title"],
            "text": chunk["text"],
            "pmid": chunk["pmid"],
            "hybrid_score": presentation_score,
            "is_paper": True
        })
        if len(merged_results) >= 5:
            break

    print(f"[RAG Ingestion Node] Successfully appended {len(merged_results)} hybrid-ranked papers to state payload.")
    print(f"[RAG Ingestion Node] Successfully extracted {len(kg_substructure)} graph cached medical pathways.")

    test_harness_chunks = list(merged_results)
    for edge in kg_substructure:
        test_harness_chunks.append({
            "title": f"Graph Edge: {edge.get('source')} -> {edge.get('target')}",
            "text": f"Relationship type: {edge.get('type')}",
            "hybrid_score": 0.0,
            "is_paper": False
        })
    return {
        "retrieved_chunks": test_harness_chunks,
        "graph_knowledge": kg_substructure
    }