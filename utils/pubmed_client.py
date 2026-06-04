import os
import requests
import urllib.parse
import time
import xml.etree.ElementTree as ET

# 🔑 Pull authenticated credentials cleanly from your local .env context
NCBI_API_KEY = os.getenv("NCBI_API_KEY")

def search_pubmed_pcos(query_string: str, max_results: int = 5):
    """Searches PubMed strictly via live network retrieval using lightweight abstract

    parameters and auto-retry loops to stabilize slow NCBI server responses.
    """
    base_search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    base_fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    # 1. URL-encode the structured search query
    safe_query = urllib.parse.quote(query_string)
    search_params = f"?db=pubmed&term={safe_query}&retmode=json&retmax={max_results}"
    
    if NCBI_API_KEY:
        search_params += f"&api_key={NCBI_API_KEY}"
        
    # 🔄 Live Connection Retry Loop (Up to 3 Retries for E-Utilities Fluctuations)
    response = None
    for attempt in range(1, 4):
        try:
            print(f"[PubMed Network] Search API - Attempt {attempt}/3...")
            response = requests.get(base_search_url + search_params, timeout=10)
            if response.status_code == 200:
                break
            else:
                print(f"[PubMed Network Warning] Search returned status code {response.status_code}. Retrying...")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as net_err:
            print(f"[PubMed Network Warning] Search link stuttered: {net_err}")
            if attempt < 3:
                time.sleep(2)
            else:
                print("❌ CRITICAL: Live Search API connection failed after 3 attempts.")
                raise net_err

    if not response or response.status_code != 200:
        return []

    try:
        # Extract Paper UIDs from the live JSON response
        data = response.json()
        id_list = data.get("esearchresult", {}).get("idlist", [])
        
        if not id_list:
            print("[PubMed Live] Active connection verified. 0 documents matched criteria.")
            return []
            
        # 🚀 OPTIMIZATION: Scope parameters strictly to 'abstract' to drop massive publisher metadata tables
        id_str = ",".join(id_list)
        fetch_url = f"{base_fetch_url}?db=pubmed&id={id_str}&retmode=xml&rettype=abstract"
        if NCBI_API_KEY:
            fetch_url += f"&api_key={NCBI_API_KEY}"
            
        print(f"[PubMed Network] Fetching lightweight XML abstracts for UIDs: {id_list}...")
        
        # 🔄 Live Fetch Connection Retry Loop (With a generous 25-second window)
        fetch_response = None
        for fetch_attempt in range(1, 4):
            try:
                fetch_response = requests.get(fetch_url, timeout=25)
                if fetch_response.status_code == 200:
                    break
                else:
                    print(f"[PubMed Network Warning] Fetch returned status code {fetch_response.status_code}. Retrying...")
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as fetch_err:
                print(f"[PubMed Network Warning] Fetch link connection timeout: {fetch_err}")
                if fetch_attempt < 3:
                    time.sleep(2)
                else:
                    print("❌ CRITICAL: Live Fetch API connection timed out after 3 attempts.")
                    raise fetch_err

        if not fetch_response or fetch_response.status_code != 200:
            print("❌ CRITICAL: Fetch server dropped active payload tracking.")
            return []
            
        # 4. ROBUST BULLETPROOF XML PARSING LAYER
        root = ET.fromstring(fetch_response.content)
        parsed_articles = []
        
        # Parse flexibly across any variant root-level citation grouping tags
        articles = root.findall(".//PubmedArticle") or root.findall(".//MedlineCitation") or root.findall(".//DocumentSummary")
        if not articles:
            articles = root.findall("*")
            
        for article in articles:
            # Extract title safely using multi-path evaluation strategies
            title_node = article.find(".//ArticleTitle") or article.find(".//Title")
            title_text = title_node.text if title_node is not None else "Indexed PubMed Study"
            
            # Extract abstract paragraphs safely across varying XML nested layers
            abstract_nodes = article.findall(".//AbstractText")
            
            abstract_text_pieces = []
            if abstract_nodes:
                for node in abstract_nodes:
                    if node.text:
                        abstract_text_pieces.append(node.text)
                    elif list(node):
                        text_parts = "".join(node.itertext()).strip()
                        if text_parts:
                            abstract_text_pieces.append(text_parts)
                            
            abstract_content = " ".join(abstract_text_pieces).strip()
            
            if not abstract_content:
                summary_node = article.find(".//Summary") or article.find(".//Abstract")
                if summary_node is not None and summary_node.text:
                    abstract_content = summary_node.text.strip()
                    
            if abstract_content:
                parsed_articles.append({
                    "title": str(title_text),
                    "text": str(abstract_content),
                    "is_paper": True
                })
                
        print(f"[PubMed Parser] Successfully retrieved and mapped {len(parsed_articles)} live abstracts from NCBI.")
        return parsed_articles

    except Exception as e:
        print(f" [PubMed Processing Error] Failed to handle live data stream: {str(e)}")
        return []