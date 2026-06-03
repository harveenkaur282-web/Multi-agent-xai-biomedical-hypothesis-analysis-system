import os
import time
from Bio import Entrez
from dotenv import load_dotenv

load_dotenv()
os.environ["XML_CATALOG_FILES"] = ""
Entrez.email = os.getenv("NCBI_EMAIL", "harveenkaur282@gmail.com")
Entrez.api_key = os.getenv("NCBI_API_KEY", None)
Entrez.TOOL = "BiomedicalHypothesisAnalysisSystem"

def search_pubmed_pcos(query, max_results=5):
    """
    Fetches raw medical abstracts from PubMed without hitting external DTD blocks.
    """
    try:
        # Your existing search logic...
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, usehistory="y")
        search_results = Entrez.read(handle)
        handle.close()
        
        # ... Your existing efetch extraction logic follows below ...
        return search_results
        
    except Exception as e:
        print(f"[PubMed API Error] standard request failed: {e}")
        # Feel free to raise the error or return a local cache here if needed
        raise e