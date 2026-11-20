"""Download the publicly released AAVAIL sample data to a local CSV.

Requires internet and an explicit --url. Review dataset licensing and provenance.
"""
import argparse
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DEFAULT_URL="https://raw.githubusercontent.com/aavail/ai-workflow-capstone-soln/master/data/aavail-target.csv"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--url",default=DEFAULT_URL)
    parser.add_argument("--output",default="data/aavail-target.csv")
    args=parser.parse_args()
    if urlparse(args.url).scheme != "https":
        parser.error("Only HTTPS sources are allowed")
    request=Request(args.url,headers={"User-Agent":"AAVAIL-Research-Capstone/1.0"})
    with urlopen(request,timeout=30) as response:
        content=response.read(4_000_000)
    if not content.startswith(b"customer_id,") or b"is_subscriber" not in content.split(b"\n",1)[0]:
        parser.error("Unexpected CSV header; refuse to overwrite destination")
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(content)
    print(f"Saved {len(content)} bytes to {dest}. Verify authenticity before use.")


if __name__=="__main__":main()
