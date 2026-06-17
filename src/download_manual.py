import os
import requests

def download_pdf():
    pdf_url = "https://euroec.by/assets/files/siemens/s71200_easy_book_en-US_en-US.pdf"
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    target_path = os.path.join(target_dir, "s71200_easy_book.pdf")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")

    if os.path.exists(target_path):
        print(f"File already exists at: {target_path}. Skipping download.")
        return target_path

    print(f"Downloading Siemens SIMATIC S7-1200 Easy Book PDF...")
    print(f"Source URL: {pdf_url}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(pdf_url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"Progress: {percent:.2f}% ({downloaded}/{total_size} bytes)", end="\r")
        
        print(f"\nDownload completed successfully! Saved to: {target_path}")
        return target_path
    except Exception as e:
        print(f"\nError downloading PDF: {e}")
        print("Please download the PDF manually and place it at:")
        print(target_path)
        return None

if __name__ == "__main__":
    download_pdf()
