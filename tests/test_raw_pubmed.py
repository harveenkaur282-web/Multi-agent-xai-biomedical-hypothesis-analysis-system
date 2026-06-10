import urllib.parse
import requests

query = '("PCOS" OR "polycystic ovary syndrome") AND ("polycystic ovary morphology" OR "insulin" OR "oligomenorrhea" OR "hirsutism" OR "acne" OR "lh-fsh ratio")'
safe_query = urllib.parse.quote(query)

url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={safe_query}&retmode=json&retmax=5"

print("[PubMed Network] Pinging PubMed directly via raw requests...")
res = requests.get(url)
print(f"Status: {res.status_code}")
print(f"Payload Response: {res.text}")