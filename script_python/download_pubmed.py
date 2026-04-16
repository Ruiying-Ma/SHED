import requests
from xml.etree import ElementTree
import time
import os

# =====================
# CONFIGURATION
# =====================
query = "machine learning"
num_articles = 5
output_dir = "PMC_articles"
os.makedirs(output_dir, exist_ok=True)

HEADERS = {
    "User-Agent": "Python script for research (your_email@example.com)"
}

# =====================
# HELPER FUNCTION: Fetch PubMed PMIDs
# =====================
def fetch_pubmed_pmids(query, retmax=5, retries=3):
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": query, "retmax": retmax, "retmode": "xml"}

    for attempt in range(retries):
        try:
            response = requests.get(search_url, params=params, headers=HEADERS, timeout=10)
            response.raise_for_status()
            root = ElementTree.fromstring(response.text)
            pmids = [id_elem.text for id_elem in root.findall(".//Id")]
            return pmids
        except requests.exceptions.RequestException as e:
            print(f"[PubMed search] Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    print("Failed to fetch PMIDs.")
    return []

# =====================
# HELPER FUNCTION: Get PMC IDs from PMIDs
# =====================
def fetch_pmc_ids(pmids, retries=3):
    pmc_ids = []
    link_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    
    for pmid in pmids:
        params = {"dbfrom": "pubmed", "db": "pmc", "id": pmid, "retmode": "xml"}
        for attempt in range(retries):
            try:
                response = requests.get(link_url, params=params, headers=HEADERS, timeout=10)
                response.raise_for_status()
                root = ElementTree.fromstring(response.text)
                # Extract PMC ID
                for linksetdb in root.findall(".//LinkSetDb"):
                    if linksetdb.find("LinkName").text == "pubmed_pmc":
                        for link in linksetdb.findall("Link/Id"):
                            pmc_ids.append(link.text)
                break
            except requests.exceptions.RequestException as e:
                print(f"[PMC link] Attempt {attempt+1} failed for PMID {pmid}: {e}")
                time.sleep(2)
        time.sleep(0.3)  # Politeness delay
    return pmc_ids

# =====================
# HELPER FUNCTION: Fetch full-text XML from PMC
# =====================
def fetch_pmc_fulltext(pmc_id, retries=3):
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/xml/"
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                return response.text
            else:
                print(f"[PMC fetch] PMC{pmc_id} returned status {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"[PMC fetch] Attempt {attempt+1} failed for PMC{pmc_id}: {e}")
            time.sleep(2)
    return None

# =====================
# MAIN SCRIPT
# =====================
print("Searching PubMed...")
pmids = fetch_pubmed_pmids(query, retmax=num_articles)
print(f"Found PMIDs: {pmids}")

if not pmids:
    exit("No PMIDs found. Exiting.")

print("Fetching PMC IDs...")
pmc_ids = fetch_pmc_ids(pmids)
print(f"Found PMC IDs: {pmc_ids}")

if not pmc_ids:
    exit("No open-access PMC full text available. Exiting.")

print("Downloading full-text XMLs...")
for pmc_id in pmc_ids:
    xml_text = fetch_pmc_fulltext(pmc_id)
    if xml_text:
        filename = os.path.join(output_dir, f"PMC{pmc_id}.xml")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(xml_text)
        print(f"Saved PMC{pmc_id} to {filename}")
    time.sleep(0.5)  # Politeness delay

print("Done!")
