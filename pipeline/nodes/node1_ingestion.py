import re
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pipeline.state import PCOSState
from utils.pubmed_client import search_pubmed_pcos

def tokenize(text: str) -> list:
    """
    Advanced regex tokenizer that handles hyphens, punctuation, and medical terms 
    much better than a naive space split.
    """
    return re.findall(r'\b[a-z0-9]+\b', text.lower())

def extract_biomedical_entities(text: str) -> list:
    """
    Lightweight Clinical Entity Recognition (NER) mapping colloquial 
    symptoms and presentations to formal MeSH/UMLS concepts.
    """
    text_lower = text.lower()
    discovered_entities = []
    
    biomedical_ontology = {
        "hirsutism": ["hirsutism", "excess hair", "facial hair", "male pattern hair", "body hair"],
        "hyperandrogenism": ["androgen", "testosterone", "acne", "oily skin", "sebum"],
        "insulin resistance": ["insulin resistance", "high insulin", "fasting insulin", "homa-ir", "metabolic syndrome"],
        "oligomenorrhea": ["oligomenorrhea", "irregular periods", "missed periods", "absent cycles", "amenorrhea"],
        "polycystic ovary morphology": ["string of pearls", "polycystic", "ovarian cysts", "follicles", "enlarged ovary"],
        "anovulation": ["anovulation", "infertility", "ovulation failure", "conception struggles"],
        "hyperglycemia": ["high sugar", "glucose intolerance", "prediabetes", "blood sugar"]
    }
    
    for formal_entity, colloquial_triggers in biomedical_ontology.items():
        for trigger in colloquial_triggers:
            if trigger in text_lower:
                discovered_entities.append(formal_entity)
                break 
                
    return list(set(discovered_entities))

def node1_ingestion_fn(state: PCOSState) -> dict:
    raw_case = state["raw_input"]
    
    # UNIFIED TEXT AGGREGATION LAYER (SCANNING ALL FIELDS)
    text_sources = []
    
    remarks = raw_case.get("clinical_remarks", "")
    if remarks:
        text_sources.append(str(remarks))
        
    symptoms = raw_case.get("symptoms", [])
    if isinstance(symptoms, list):
        text_sources.extend([str(s) for s in symptoms])
    elif isinstance(symptoms, str) and symptoms:
        text_sources.append(symptoms)
        
    user_query = raw_case.get("query", "")
    if user_query:
        text_sources.append(str(user_query))
        
    ultrasound = raw_case.get("ultrasound", "")
    if ultrasound:
        text_sources.append(str(ultrasound).replace("_", " "))
        
    combined_text = " ".join(text_sources).strip()
    extracted_concepts = extract_biomedical_entities(combined_text)
    
    # DYNAMIC QUERY FORMULATION LAYER
    query_tokens = ["PCOS", "polycystic ovary syndrome"]
    if extracted_concepts:
        query_tokens.extend(extracted_concepts)
    else:
        query_tokens.append("insulin resistance metabolic endocrine dysregulation")
        
    # ──── FIX 2: SAFE DEFAULT PARAMETER FALLBACKS (REMOVED SILENT INJECTIONS) ────
    if float(raw_case.get("fasting_insulin", 0.0)) >= 14.0:
        query_tokens.append("hyperinsulinemia")
    if float(raw_case.get("lh_fsh_ratio", 0.0)) >= 2.0:
        query_tokens.append("lh-fsh ratio inversion")
    # ────────────────────────────────────────────────────────────────────────────
        
    query = " ".join(query_tokens).strip()
    print(f"[RAG Ingestion] Formulated MeSH Query: '{query}'")
    
    # API DATABASE RETRIEVAL
    raw_api_response = search_pubmed_pcos(query)
    
    articles = []
    if isinstance(raw_api_response, dict) and "PubmedArticle" in raw_api_response:
        articles = raw_api_response["PubmedArticle"]
    elif isinstance(raw_api_response, list):
        articles = raw_api_response
    else:
        articles = [raw_api_response] if raw_api_response else []

    if not articles:
        print("[Ingestion Warning] No medical literature matched entity signatures.")
        return {"retrieved_chunks": []}

    # Extract Title and Abstract safely
    documents = []
    for idx, article in enumerate(articles):
        try:
            citation = article.get("MedlineCitation", {})
            inner_article = citation.get("Article", {})
            title = inner_article.get("ArticleTitle", f"PubMed Study Reference {idx+1}")
            
            abstract_obj = inner_article.get("Abstract", {})
            abstract_text_list = abstract_obj.get("AbstractText", [""])
            text = " ".join([str(t) for t in abstract_text_list]) if abstract_text_list else ""
            
            if not text: 
                continue
                
            documents.append({"title": str(title), "text": str(text)})
        except Exception:
            if isinstance(article, dict):
                documents.append({
                    "title": article.get("title", f"PubMed Clip {idx+1}"), 
                    "text": article.get("text", article.get("abstract", ""))
                })

    if not documents:
        return {"retrieved_chunks": []}

    # ADVANCED REGEX BM25 TOKENIZATION PROCESS
    corpus = [doc["text"] for doc in documents]
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenize(query))
    
    # SAFE NORMALIZATION FOR SINGLE-ARTICLE / IDENTICAL SCORES
    score_min, score_max = np.min(bm25_scores), np.max(bm25_scores)
    if len(bm25_scores) == 1 or (score_max - score_min) == 0:
        normalized_bm25 = np.ones(len(bm25_scores)) * 0.5
    else:
        normalized_bm25 = (bm25_scores - score_min) / (score_max - score_min)
        
    # ──── FIX 1: REMOVED stop_words='english' TO SILENCE USER WARNINGS ────
    vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words=None)
    tfidf_matrix = vectorizer.fit_transform([query] + corpus)
    cosine_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    # ──────────────────────────────────────────────────────────────────────
    
    # Assemble sorted result dictionary payload using explicit architecture weights
    merged_results = []
    for idx, doc in enumerate(documents):
        semantic_score = float(cosine_scores[idx])
        # 40% Lexical matching + 60% Semantic matching
        combined_score = float((0.4 * normalized_bm25[idx]) + (0.6 * semantic_score))
        
        merged_results.append({
            "title": doc["title"],
            "text": doc["text"],
            "hybrid_score": round(combined_score, 4)
        })
        
    merged_results = sorted(merged_results, key=lambda x: x["hybrid_score"], reverse=True)[:10]
    return {"retrieved_chunks": merged_results}