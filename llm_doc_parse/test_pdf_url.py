import requests

url = "https://raw.githubusercontent.com/Ruiying-Ma/SHTRAG/main/data/civic/pdf/01272021-1626.pdf"
output_path = "01272021-1626.pdf"

response = requests.get(url)
response.raise_for_status()  # raise error if request failed

with open(output_path, "wb") as f:
    f.write(response.content)

print(f"Downloaded PDF to {output_path}")
